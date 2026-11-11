"""!
@package download_gsv_panos
@brief Street View metadata and imagery (four 90-degree frames per node).

Steps (in this order):
  1. metadata   which panorama serves each node (probe; free), with its capture date, position and photographer, cached
                in street-view-metadata.csv (metadata); only unanswered nodes and user uploads are re-asked. Where
                the nearest panorama is a user upload, points a few metres along the street are asked for Google's
  2. selection  which nodes to download (select): covered, panorama within max_pano_offset_m of the node; user uploads
                too (marked user_upload, unusable unless node-review-decisions.csv keeps them). Any capture date is
                accepted; it is recorded as pano_date. Why a node is not usable (why_not)
  3. imagery    four frames per node, by pano_id (never by coordinate), so all headings come from the same panorama;
                existing files are kept (download)
  4. run        writes street-view-frames.csv, records downloaded / reason per node in street-view-metadata.csv;
                both go in the download log

Google caps caching Street View imagery at 30 days: treat data/<city>/data_prep/downloads/street-view-frames/ as
scratch. Run by pipeline step download_street_view (the API key: GMAPS_KEY in .env); stitch_180_views.py
turns the frames into the 180-degree views (stitch_180_views).
"""
import math
import sys

import geopandas as gpd
import pandas as pd
import requests
from dotenv import dotenv_values
from tqdm.auto import tqdm

from common.downloads import record

# Google Street View Static API (key GMAPS_KEY in .env): Google Maps Platform terms; imagery may be cached for 30
# days only, never redistributed
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
FINAL = {"OK", "ZERO_RESULTS", "NOT_FOUND"}   # metadata answers worth caching; others are re-asked
SHIFTS_M = (0, 3, -3, 6, -6, 10, -10)        # probe points along the street (m from the node), until Google answers


# ---------------------------------------------------------------- 1. metadata
def probe(nodes, key):
    """!
    @brief Ask Street View which panorama serves each node (metadata requests are free).

    The nearest panorama can be a user upload (source=outdoor still returns them); then points SHIFTS_M along
    the street are asked, and the first Google panorama is taken. None found: the node's own answer.

    @param nodes  DataFrame with node_id, lat, lon, heading_fwd_deg.
    @param key    API key.
    @return DataFrame, one row per node: status, pano_id, pano_date (YYYY-MM), copyright
            ("© Google" for Google's captures, the uploader otherwise), pano and node lat/lon, probe_shift_m.
    """
    rows = []
    for r in tqdm(list(nodes.itertuples()), desc="metadata", mininterval=2.0):
        answers = []
        for shift in SHIFTS_M:
            h = math.radians(r.heading_fwd_deg)                 # along the street, metres to degrees
            lat = r.lat + shift * math.cos(h) / 111320
            lon = r.lon + shift * math.sin(h) / (111320 * math.cos(math.radians(r.lat)))
            params = {"location": f"{lat},{lon}", "source": "outdoor", "key": key}
            try:
                js = requests.get(METADATA_URL, params=params, timeout=15).json()
            except (requests.RequestException, ValueError) as ex:   # network error or a non-JSON reply
                js = {"status": f"ERROR {ex}"}
            answers.append((shift, js))
            if "Google" in str(js.get("copyright")):
                break                                          # Google's panorama; else (upload, error) keep looking
        shift, js = next((a for a in answers if "Google" in str(a[1].get("copyright"))), answers[0])
        loc = js.get("location") or {}
        rows.append({"node_id": r.node_id, "status": js.get("status"),
                     "pano_id": js.get("pano_id"), "pano_date": js.get("date"),
                     "copyright": js.get("copyright"),
                     "pano_lat": loc.get("lat"), "pano_lon": loc.get("lng"),
                     "node_lat": r.lat, "node_lon": r.lon, "probe_shift_m": shift})
    return pd.DataFrame(rows)


def metadata(nodes, key, cached_csv, crs):
    """!
    @brief Metadata for every node: cached answers reused, the rest probed.

    A cached row is reused only if it belongs to these nodes and its status is a
    final answer, so a bad key, a quota error or a network failure is retried
    on the next run instead of being cached as "no coverage".

    @param nodes       DataFrame with node_id, lat, lon.
    @param key         API key.
    @param cached_csv  street-view-metadata.csv from an earlier run (may not exist).
    @param crs         A metre CRS, for the node-to-panorama distance.
    @return DataFrame from probe(), plus pano_offset_m (node to panorama, metres) and user_upload.
    """
    cached = pd.read_csv(cached_csv) if cached_csv.exists() else pd.DataFrame(columns=["node_id", "status"])
    upload = cached.status.eq("OK") & ~cached.get("copyright", pd.Series(dtype=str)).astype(str).str.contains("Google")
    cached = cached[cached.node_id.isin(nodes.node_id) & cached.status.isin(FINAL) & ~upload]   # uploads re-asked
    todo = nodes[~nodes.node_id.isin(cached.node_id)]
    print(f"metadata: {len(cached)} cached, {len(todo)} to ask")

    meta = pd.concat([cached, probe(todo, key)], ignore_index=True) if len(todo) else cached.reset_index(drop=True)
    meta = meta.drop(columns=[c for c in ("pano_offset_m", "user_upload", "fetch", "downloaded", "reason")
                              if c in meta])                                   # recomputed below
    meta["probe_shift_m"] = meta.get("probe_shift_m", pd.Series(0.0, index=meta.index)).fillna(0)  # older rows: 0

    # distance from node to panorama, in metres
    meta["pano_offset_m"] = float("nan")
    has = meta.pano_lat.notna()
    pano = gpd.GeoSeries(gpd.points_from_xy(meta.pano_lon[has], meta.pano_lat[has]), crs=4326).to_crs(crs)
    node = gpd.GeoSeries(gpd.points_from_xy(meta.node_lon[has], meta.node_lat[has]), crs=4326).to_crs(crs)
    meta.loc[has, "pano_offset_m"] = pano.distance(node, align=False).values
    # Google's own captures carry "© Google"; anything else was uploaded by a user (bus or shop interiors, ...)
    meta["user_upload"] = meta.status.eq("OK") & ~meta.copyright.astype(str).str.contains("Google")
    return meta


# ---------------------------------------------------------------- 2. selection
def select(meta, SP):
    """!
    @brief Mark the nodes to download (column `fetch`) and print the funnel; capture date is not a filter.
    @param meta  DataFrame from metadata().
    @param SP    The config's `download_gsv_panos`.
    @return The nodes to download (rows of `meta`).
    """
    ok = meta.status.eq("OK")                                               # Street View covers it
    near = meta.pano_offset_m.le(SP["max_pano_offset_m"])                   # panorama close to the node
    meta["fetch"] = ok & near                                               # user uploads too: they are reviewed

    print(f"  nodes {len(meta)}   with coverage {int(ok.sum())}   pano within {SP['max_pano_offset_m']} m "
          f"{int(meta.fetch.sum())} (user uploads {int((meta.fetch & meta.user_upload).sum())})")
    print("  capture dates of those: " + ", ".join(
        f"{d} ×{k}" for d, k in meta[meta.fetch].pano_date.value_counts().sort_index().items()))
    if not meta.fetch.any():
        denied = (meta.status == "REQUEST_DENIED").any()
        sys.exit("no usable nodes" + ("; REQUEST_DENIED: check the key and billing" if denied else ""))
    return meta[meta.fetch]


def why_not(r, SP):
    """!
    @brief Why a node's imagery is not usable, from its metadata row ("" if it is).
    @param r   A row of metadata() with `downloaded`.
    @param SP  The config's `download_gsv_panos`.
    @return The first reason that applies. A user upload is downloaded (for review) but not usable unless
            node-review-decisions.csv says keep.
    """
    if r.status != "OK":
        return f"no Street View panorama ({r.status})"
    if not r.pano_offset_m <= SP["max_pano_offset_m"]:
        return f"panorama {r.pano_offset_m:.0f} m from the node (max {SP['max_pano_offset_m']} m)"
    if r.user_upload:
        return f"user-uploaded panorama ({r.copyright})"
    if not r.downloaded:
        return "imagery download failed (rerun download_street_view)"
    return ""


# ---------------------------------------------------------------- 3. imagery
def download(use, key, SP, folder):
    """!
    @brief Four frames per node, requested by pano_id; files already on disk are kept.
    @param use     Rows of metadata() to download.
    @param key     API key.
    @param SP      The config's `download_gsv_panos`.
    @param folder  Where the frames go (street-view-frames/).
    @return (manifest DataFrame: node_id, heading, file (in `folder`), pano_date, user_upload; number of failed
            requests).
    """
    folder.mkdir(parents=True, exist_ok=True)
    todo = sum(not (folder / f"{r.node_id}_{h:03d}.jpg").exists() for r in use.itertuples() for h in SP["headings"])
    print(f"imagery: {todo} request(s) to make (${todo * 0.007:.2f} at list price)")

    rows, failed = [], 0
    for r in tqdm(list(use.itertuples()), desc="imagery", mininterval=2.0):
        for h in SP["headings"]:
            path = folder / f"{r.node_id}_{h:03d}.jpg"
            if not path.exists():
                params = {"pano": r.pano_id, "size": f"{SP['image_size']}x{SP['image_size']}",
                          "heading": h, "fov": SP["fov"], "pitch": SP["pitch"],
                          "return_error_code": "true",   # a 404, not a grey "no imagery" picture
                          "key": key}
                try:
                    resp = requests.get(IMAGE_URL, params=params, timeout=30)
                    resp.raise_for_status()
                except requests.RequestException:
                    failed += 1
                    continue
                path.write_bytes(resp.content)
            rows.append({"node_id": r.node_id, "heading": h, "file": path.name,
                         "pano_date": r.pano_date, "user_upload": bool(r.user_upload)})
    return pd.DataFrame(rows), failed


# ---------------------------------------------------------------- 4. run
def run(cfg, paths):
    """!
    @brief Metadata, selection, imagery; writes street-view-metadata.csv (with downloaded / reason) and
           street-view-frames.csv, both recorded in the download log. The key is GMAPS_KEY in .env.
    """
    SP = cfg["download_gsv_panos"]
    env = paths.root / ".env"                                # git-ignored; the key's one source
    key = dotenv_values(env).get("GMAPS_KEY") or sys.exit(f"GMAPS_KEY not set: add GMAPS_KEY=... to {env}")
    nodes = gpd.read_file(paths.nodes_gpkg)

    # 1. which panorama serves each node
    was = pd.read_csv(paths.pano_metadata_csv) if paths.pano_metadata_csv.exists() else None
    meta = metadata(nodes, key, paths.pano_metadata_csv, int(cfg["projected_crs"]))
    meta.to_csv(paths.pano_metadata_csv, index=False)   # saved before select() can stop the run

    # frames are named by node, not panorama: a node whose panorama changed has its old frames removed
    if was is not None:
        old = meta.node_id.map(was.set_index("node_id").pano_id)
        for n in meta.node_id[old.notna() & old.ne(meta.pano_id)]:
            for p in paths.raw_photos.glob(f"{n}_[0-9][0-9][0-9].jpg"):
                p.unlink()

    # 2. which nodes to download
    use = select(meta, SP)

    # images left from other nodes would be paired with the wrong nodes
    stale = [p for p in paths.raw_photos.glob("*.jpg") if p.stem.rsplit("_", 1)[0] not in set(nodes.node_id)]
    if stale:
        sys.exit(f"{len(stale)} images in {paths.raw_photos} belong to other nodes. Delete {paths.raw_photos} "
                 f"and re-run.")

    # 3. imagery, and the manifest later steps read
    manifest, failed = download(use, key, SP, paths.raw_photos)
    if manifest.empty:
        sys.exit("no imagery downloaded")
    manifest.to_csv(paths.raw_photos_csv, index=False)
    print(f"{len(manifest)} images across {manifest.node_id.nunique()} nodes, {failed} failed")

    # the decision per node, next to the facts it rests on: read_nodes() reads `downloaded` and `reason`
    meta["downloaded"] = meta.node_id.map(manifest.groupby("node_id").size()).eq(len(SP["headings"]))
    meta["reason"] = [why_not(r, SP) for r in meta.itertuples()]
    meta.to_csv(paths.pano_metadata_csv, index=False)
    record(paths, "google_street_view", paths.pano_metadata_csv, METADATA_URL)
    record(paths, "google_street_view", paths.raw_photos_csv, IMAGE_URL)
    reasons = meta.loc[meta.reason != "", "reason"].str.split(" [(]").str[0].value_counts()
    print(reasons.to_string() if len(reasons) else "every node usable")
