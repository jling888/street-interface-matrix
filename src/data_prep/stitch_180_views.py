"""!
@package stitch_180_views
@brief Two 180-degree views per node, one for each walking direction, stitched from its four Street View frames.

Steps (in this order):
  1. walks     each street's two walks: their folder names (walk_name), the nodes in walking order with their
               headings (plan), and a compass letter for file names (cardinal)
  2. panorama  a node's four frames (load_frames), reprojected into one cylindrical 180-degree strip centred on a
               heading (panorama; bilinear sampling)
  3. run       renders every node with a complete frame set, writes views-180.csv

Headings and order come from the nodes (heading_fwd_deg / heading_rev_deg, seq_fwd / seq_rev),
so curved streets render along their own line. Unusable nodes are rendered too (the
review app shows them); the Street Interface Matrix drops their ratings. Everything is written to
data/<city>/data_prep/: views-180/<street>/<walk>/<seq>_<node>_<N|E|S|W>.jpg and
views-180.csv, the index the VLM rating, the scoring and the review app read.

Run by pipeline step stitch_180_views.
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import map_coordinates
from tqdm.auto import tqdm


# ---------------------------------------------------------------- 1. walks
def walk_name(bearing):
    """!
    @brief Folder name for walking at `bearing`: the dominant compass direction.
    @param bearing  Degrees, 0 = north, clockwise.
    @return "south_to_north", "north_to_south", "west_to_east" or "east_to_west".
    """
    east, north = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
    if abs(east) > abs(north):
        return "west_to_east" if east > 0 else "east_to_west"
    return "south_to_north" if north > 0 else "north_to_south"


def cardinal(bearing):
    """!
    @brief Nearest compass letter, for file names.
    @param bearing  Degrees.
    @return "N", "E", "S" or "W".
    """
    return "NESW"[int(round(bearing / 90)) % 4]


def plan(street):
    """!
    @brief A street's two walks.
    @param street  Its rows of nodes.csv.
    @return [(walk folder, [(node_id, heading), ...] in walking order)] for the forward and reverse walk.
    """
    walks = []
    for heading, seq in (("heading_fwd_deg", "seq_fwd"), ("heading_rev_deg", "seq_rev")):
        o = street.sort_values(seq)
        walks.append((walk_name(o[heading].iloc[0]), list(zip(o.node_id, o[heading].astype(float)))))
    return walks


# ---------------------------------------------------------------- 2. panorama
def load_frames(rows, paths, n_headings):
    """!
    @brief A node's frames as arrays, keyed by heading.
    @param rows        The node's rows of street-view-frames.csv.
    @param n_headings  How many frames a node has (download_gsv_panos.headings).
    @return {heading: HxWx3 uint8}, or None unless every configured heading opens (a gap would read as
            a dark street).
    """
    frames = {}
    for f in rows.itertuples():
        try:
            frames[float(f.heading)] = np.asarray(Image.open(paths.raw_photos / f.file).convert("RGB"))
        except OSError:
            return None
    return frames if len(frames) == n_headings else None


def panorama(frames, heading, width, frame_fov, strip_fov):
    """!
    @brief Reproject perspective frames into one cylindrical strip of `strip_fov` degrees centred on `heading`.

    Each output column takes the frame whose heading is closest; within it the pixel position follows
    the gnomonic (perspective) projection, sampled bilinearly. Rows cover the frames' vertical field of view
    (+-45 degrees of elevation for square 90-degree frames).

    @param frames     {heading: HxWx3 array}, each with the field of view `frame_fov` (download_gsv_panos.fov).
    @param heading    Centre bearing of the strip, degrees.
    @param width      Output width in pixels.
    @param strip_fov  The strip's horizontal field of view (stitch_180_views.fov).
    @return HxWx3 uint8 strip.
    """
    H, W = next(iter(frames.values())).shape[:2]
    f = W / 2 / np.tan(np.radians(frame_fov / 2))                     # frame focal length, pixels
    fc = width / np.radians(strip_fov)                                # strip: pixels per radian
    height = int(2 * fc * (np.tan(np.radians(frame_fov / 2)) * (H / W)))   # tan of the frames' top elevation
    theta = np.radians(heading) + (np.arange(width) - width / 2) / fc  # bearing of each column
    phi = np.arctan((height / 2 - np.arange(height)) / fc)            # elevation of each row (row 0 = top)

    heads = np.array(sorted(frames))
    dh = (np.degrees(theta)[:, None] - heads[None, :] + 180) % 360 - 180
    pick = np.abs(dh).argmin(axis=1)                                  # nearest frame per column
    alpha = np.radians(dh[np.arange(width), pick])                    # angle off that frame's centre

    out = np.zeros((height, width, 3), dtype=np.uint8)
    for k, h in enumerate(heads):
        col = np.flatnonzero(pick == k)
        if not len(col):
            continue
        x = f * np.tan(alpha[col]) + W / 2 - 0.5                      # source pixel coordinates
        y = H / 2 - f * np.tan(phi)[:, None] / np.cos(alpha[col])[None, :] - 0.5
        coords = [y, np.broadcast_to(x, y.shape)]
        for c in range(3):
            out[:, col, c] = map_coordinates(frames[h][..., c], coords, order=1, mode="nearest")
    return out


# ---------------------------------------------------------------- 3. run
def run(cfg, paths):
    """!
    @brief Render both walks of every street, write views-180.csv.
    """
    SP, views = cfg["download_gsv_panos"], cfg["stitch_180_views"]     # the frames, the views

    # the nodes with downloaded imagery, by street
    manifest = pd.read_csv(paths.raw_photos_csv)
    nodes = pd.read_csv(paths.nodes_csv)
    nodes = nodes[nodes.node_id.isin(manifest.node_id)]               # nodes with imagery
    streets = sorted(nodes.street.unique())
    print(f"{len(nodes)} nodes, {len(streets)} streets -> {paths.views}")

    by_node = {n: g for n, g in manifest.groupby("node_id")}
    rows, missing = [], set()
    for street in tqdm(streets, desc="streets", mininterval=1.0):
        # 1. the street's two walks, nodes in walking order, each in a folder named for its direction
        for walk, seq in plan(nodes[nodes.street == street]):
            folder = paths.views / street / walk
            folder.mkdir(parents=True, exist_ok=True)
            for i, (nid, heading) in enumerate(seq, start=1):
                # 2. one 180-degree strip along the node's heading (a node missing a frame is
                #    skipped: the gap would read as a dark street)
                frames = load_frames(by_node[nid], paths, len(SP["headings"]))
                if frames is None:
                    missing.add(nid)
                    continue
                name = f"{i:03d}_{nid}_{cardinal(heading)}.jpg"
                Image.fromarray(panorama(frames, heading, views["width"], SP["fov"], views["fov"])).save(
                    folder / name, quality=views["quality"], optimize=True)
                rows.append({"file": f"{street}/{walk}/{name}", "node_id": nid, "street": street, "walk": walk,
                             "seq": i, "cardinal": cardinal(heading), "bearing": round(heading % 360, 3)})

    index = pd.DataFrame(rows)
    path = paths.views_csv
    index.to_csv(path, index=False)
    print(f"wrote {len(rows)} views; index {path}")
    if missing:
        print(f"{len(missing)} node(s) skipped for an incomplete frame set")
