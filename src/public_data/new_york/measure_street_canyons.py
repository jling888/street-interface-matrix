"""!
@package measure_street_canyons
@brief Building height H, facade-to-facade street width W, H/W and the canyon's sky view factor per node.

Steps, in this order:
  1. footprints  New York's building footprints with height_m (FOOTPRINTS), downloaded the first time; also written
                 as buildings-with-height.gpkg, the buildings any later stage reads
  2. height      H = the median footprint height within corridor_m of the node
  3. width       a band probe each side of the street axis: hw_band_rays parallel rays across one node
                 spacing, the nearest wall within facade_half_m; W = the two sides' sum. A side with no wall
                 even under a wider fan (facade_open_test_deg, facade_open_reach_m) is open
  4. fill        a node with one wall and one open side keeps no ratio (open_one_side), unless the open side
                 points down a nearby cross street (a crossing); other nodes without W inherit from measured
                 nodes on the same street: the mean within hw_fill_radius_m (radius_mean), else the nearest
                 along the street, at most hw_series_max_steps nodes away (series)
  5. run         runs them on the nodes with view indices, adds SVF_canyon = cos(arctan(2 H/W)) (the sky view
                 factor at the floor of a long symmetric canyon, from HW_effective), and prints VEI against the
                 measured H/W

Runs on CPU in seconds. Skipped without view-indices-per-node.csv. Writes
data/<city>/public_data/street-canyons-per-node.csv (node_id + OWN). The Street Interface
Matrix copies these columns into its tables; no formula uses them: they validate the VLM's enclosure rating. Run by
pipeline step measure_street_canyons.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.stats import spearmanr

from common.downloads import record, socrata_shapes, socrata_url
from common.node_table import node_table, street_axis

OWN = ("H_m", "W_facade", "HW_facade", "HW_effective", "HW_source", "SVF_canyon",
       "facade_sides_hit", "facade_sides_open", "facade_empty_bearing")      # the columns this step writes


# ---------------------------------------------------------------- 1. footprints
# New York's building footprints (NYC Open Data, Building Footprints; free to use under its terms). NYC republishes
# them under new dataset ids, so the first that answers is used. Roof heights are in feet; a building's number,
# construction year and tax lot are also read by the city's records (measure_city_records).
FOOTPRINTS = {"portal": "data.cityofnewyork.us", "datasets": ["5zhs-2jue", "qb5r-6dgf", "nqwf-w8eh"],
              "file": "building-footprints.geojson", "geometry_field": "the_geom", "margin_deg": 0.001,
              "height_field": ["height_roof", "heightroof"],      # the first that exists (the datasets name it differently)
              "height_to_m": 0.3048, "height_min_m": 3, "height_max_m": 400,   # heights outside this range left out
              "bin_field": "bin", "year_field": "construction_year", "bbl_field": "mappluto_bbl"}


def footprints_file(paths):
    """!
    @brief New York's footprints file, downloaded around the nodes the first time and logged.
    """
    F, path = FOOTPRINTS, paths.public_data_downloads / FOOTPRINTS["file"]
    urls = [socrata_url(F["portal"], d) for d in F["datasets"]]
    if not path.exists():
        urls[0] = socrata_shapes(urls, F, gpd.read_file(paths.nodes_gpkg), path)
    record(paths, "building_footprints", path, urls[0])
    return path


def buildings_with_height(cfg, paths):
    """!
    @brief The footprints in the projected CRS with height_m (height_field, the first that exists, in metres) within
           height_min_m-height_max_m.
    @return GeoDataFrame (height_m, geometry).
    """
    F = FOOTPRINTS
    bf = gpd.read_file(footprints_file(paths)).to_crs(int(cfg["projected_crs"]))
    field = next((f for f in F["height_field"] if f in bf), None)
    if field is None:
        raise ValueError(f"none of the footprints' height fields {F['height_field']} is a column")
    bf["height_m"] = pd.to_numeric(bf[field], errors="coerce") * F["height_to_m"]
    return bf.loc[bf.height_m.between(F["height_min_m"], F["height_max_m"]), ["height_m", "geometry"]]


# ---------------------------------------------------------------- 2. height
def height(nodes, bf, G):
    """!
    @brief H per node: the median height of the footprints within corridor_m.
    @param G  The config's `street_canyon_geometry`.
    @return Series H_m by node_id (nodes with none in reach absent).
    """
    buf = gpd.GeoDataFrame(nodes[["node_id"]], geometry=nodes.buffer(G["corridor_m"]))
    return gpd.sjoin(bf, buf, predicate="intersects").groupby("node_id").height_m.median().rename("H_m")


# ---------------------------------------------------------------- 3. width
def reach(walls, tree, pt, bearing, width, length, rays):
    """!
    @brief Distance to the nearest wall from `pt` along `bearing`, over `rays` (hw_band_rays) parallel rays
           spread across `width`.
    @param walls  Footprint geometries (numpy array); `tree` their STRtree.
    @return Metres, or None when no ray of `length` meets a wall.
    """
    ux, uy = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
    off = np.linspace(-width / 2, width / 2, rays)
    ox, oy = pt.x + uy * off, pt.y - ux * off
    lines = shapely.linestrings(np.stack([np.c_[ox, oy], np.c_[ox + ux * length, oy + uy * length]], axis=1))
    r, w = tree.query(lines, predicate="intersects")
    if not len(r):
        return None
    # the part of a ray inside a wall starts where the ray enters it
    return float(shapely.distance(shapely.points(ox[r], oy[r]), shapely.intersection(lines[r], walls[w])).min())


def width(nodes, bf, axis, G, spacing):
    """!
    @brief W per node, facade to facade across its street axis, and which sides found a wall.
    @param G        The config's `street_canyon_geometry`.
    @param spacing  nodes.spacing_m: the band the rays span.
    @return DataFrame by node_id: facade_sides_hit, facade_sides_open, facade_empty_bearing (the open side's
            bearing), W_facade (only with walls on both sides, within hw_min_w_m-hw_max_w_m).
    """
    walls = bf.geometry.to_numpy()
    tree = shapely.STRtree(walls)
    wide = spacing * G["facade_open_test_deg"] / G["facade_cone_deg"]
    rows = []
    for r in nodes.itertuples():
        ax = axis.get(r.node_id, np.nan)
        if np.isnan(ax):
            rows.append((r.node_id, 0, 0, np.nan, np.nan))
            continue
        d, opens, empty = [], 0, np.nan
        for b in ((ax + 90) % 360, (ax - 90) % 360):
            d.append(reach(walls, tree, r.geometry, b, spacing, G["facade_half_m"], G["hw_band_rays"]))
            if d[-1] is None and reach(walls, tree, r.geometry, b, wide, G["facade_open_reach_m"],
                                       G["hw_band_rays"]) is None:
                opens, empty = opens + 1, b
        hit = [x for x in d if x is not None]
        w = sum(hit) if len(hit) == 2 and G["hw_min_w_m"] < sum(hit) < G["hw_max_w_m"] else np.nan
        rows.append((r.node_id, len(hit), opens, empty, w))
    return pd.DataFrame(rows, columns=["node_id", "facade_sides_hit", "facade_sides_open", "facade_empty_bearing",
                                       "W_facade"]).set_index("node_id")


# ---------------------------------------------------------------- 4. fill
def crossing(m, nodes, axis, G):
    """!
    @brief Of the nodes with one wall and one open side, those whose open side points along a cross street
           within open_cross_dist_m (open_cross_angle_deg): a crossing, not an open side.
    @return Boolean per row of m.
    """
    other = nodes[nodes.street.notna()]
    ox, oy, oname = other.geometry.x.to_numpy(), other.geometry.y.to_numpy(), other.street.to_numpy()
    oax = np.array([axis.get(k, np.nan) for k in other.node_id])
    out = np.zeros(len(m), bool)
    for i, r in enumerate(m.itertuples()):
        if not np.isfinite(r.facade_empty_bearing):
            continue
        far = oname != r.street
        d = np.hypot(ox[far] - r.geometry.x, oy[far] - r.geometry.y)
        if not len(d) or d.min() > G["open_cross_dist_m"]:
            continue
        ca = oax[far][d.argmin()]
        out[i] = np.isfinite(ca) and abs((r.facade_empty_bearing - ca + 90) % 180 - 90) < G["open_cross_angle_deg"]
    return out


def fill(m, nodes, axis, G):
    """!
    @brief H/W for every node: measured, open on one side, or inherited from the same street.
    @param G  The config's `street_canyon_geometry`.
    @return (HW_effective, HW_source), one per row of m.
    """
    seg, x, y = m.street.to_numpy(), m.geometry.x.to_numpy(), m.geometry.y.to_numpy()
    hw = m.HW_facade.to_numpy(float)
    filled, source = hw.copy(), np.where(np.isfinite(hw), "measured", "none").astype(object)
    open_side = (m.facade_sides_hit.to_numpy() == 1) & (m.facade_sides_open.to_numpy() == 1) & ~np.isfinite(hw)
    open_side[open_side] = ~crossing(m[open_side], nodes, axis, G)
    source[open_side] = "open_one_side"

    order = {}                                     # each street's nodes in order along its median axis
    for s in pd.unique(seg):
        k = np.flatnonzero(seg == s)
        a = np.radians(np.nanmedian([axis.get(n, np.nan) for n in m.node_id.iloc[k]]))
        order[s] = k[np.argsort(x[k] * np.sin(a) + y[k] * np.cos(a))] if np.isfinite(a) and len(k) >= 2 else k
    for i in np.flatnonzero(~np.isfinite(hw) & ~open_side):
        same = (seg == seg[i]) & np.isfinite(hw)
        near = hw[same][np.hypot(x[same] - x[i], y[same] - y[i]) <= G["hw_fill_radius_m"]]
        if len(near):
            filled[i], source[i] = near.mean(), "radius_mean"
            continue
        k = order[seg[i]]                          # walk the street outward, nearer side first
        pos = int(np.flatnonzero(k == i)[0])
        for step in range(1, G["hw_series_max_steps"] + 1):
            got = [hw[k[j]] for j in (pos - step, pos + step) if 0 <= j < len(k) and np.isfinite(hw[k[j]])]
            if got:
                filled[i], source[i] = float(np.mean(got)), "series"
                break
    return filled, source


# ---------------------------------------------------------------- 5. run
def run(cfg, paths):
    """!
    @brief H, W and H/W for every node with view indices, the gaps filled along each street.
    """
    G = cfg["street_canyon_geometry"]
    nodes = node_table(cfg, paths)
    if "VEI" not in nodes:
        print("no view-indices-per-node.csv (calculate_segmentation_pixel_shares); skipping")
        return
    bf = buildings_with_height(cfg, paths)
    paths.buildings_with_height.unlink(missing_ok=True)          # a GeoPackage takes a new layer, not a new file
    bf.to_file(paths.buildings_with_height, driver="GPKG")         # the buildings any later stage reads
    m = nodes[nodes.VEI.notna()].drop(columns=[c for c in OWN if c in nodes]).reset_index(drop=True)
    axis = street_axis(nodes)
    spacing = cfg["sample_osm_nodes"]["spacing_m"]
    m = m.join(height(m, bf, G), on="node_id").join(width(m, bf, axis, G, spacing), on="node_id")
    m["HW_facade"] = m.H_m / m.W_facade
    print(f"W measured for {m.W_facade.notna().sum()}/{len(m)} nodes")
    m["HW_effective"], m["HW_source"] = fill(m, nodes, axis, G)
    print(m.HW_source.value_counts().rename_axis(None).to_string())
    # ponytail: long symmetric canyon, sky seen from its floor centre, no trees; a photo looking up would measure it
    m["SVF_canyon"] = np.cos(np.arctan(2 * m.HW_effective))

    s = m.dropna(subset=["VEI", "HW_facade"])
    if len(s) > 2:
        print(f"VEI vs measured H/W: rho={spearmanr(s.VEI, s.HW_facade)[0]:+.3f} n={len(s)}")
    m[["node_id", *OWN]].to_csv(paths.street_canyons_csv, index=False)
    print(f"wrote {paths.street_canyons_csv}")
