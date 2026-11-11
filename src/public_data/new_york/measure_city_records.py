"""!
@package measure_city_records
@brief What the city's records and the footprints say about each node's frontage: the physical attributes of the
       Imageability framework. The VLM rates each of them too (the agreement terms of
       configs/inference/vlm_prompts.yaml), and
       step validate_vlm_ratings compares the two; the score (compose_metric) takes the record or the rating, as its
       config says.

Per node, from New York's records (SOURCES and the footprints, downloaded into data/<city>/public_data/downloads/ the
first time; the
measures in the config's `city_records_per_node`):
  frontage geometry: rays square to the street, both sides, at offsets along it, find the curb and the sidewalk's
  inner edge (planimetric sidewalks) and the first facade (footprints)
      sidewalk_width_m          the sidewalk's width, the mean of the two sides
      curb_to_facade_m          curb to the first facade, the mean of the two sides: the ground between road and
      building
      setback_m                 the deeper side's setback beyond the sidewalk
      street_wall_share         the share of rays meeting a facade within wall_m of the sidewalk (Ewing's street wall)
  the buildings fronting the node (within half the street's width plus fronting_margin_m)
      buildings_fronting, median_building_year, landmark_building_share, active_sidewalk_shed (in the photo's month)
  within radius_m of the node (the node spacing)
      street_tree_count         living street trees
      storefronts, vacant_storefront_share   the storefront filings of one reporting year
      pops_buildings, landmark_buildings     buildings with a POPS or a landmark designation, by footprint (the records
                                             are address points, which seldom lie within radius_m of a street node)
      public_art                temporary art in place in the photo's month
      cultural_anchor_count     the sum of the three
  open_space_distance_m         to the nearest park or pedestrian plaza
Writes data/<city>/public_data/city-records-per-node.csv. Run by
pipeline step measure_city_records, which first writes land use and seating (land_use_per_node).

Land use (land_use_per_node): what is on each street, from the data sources city_records_per_node.land_use names: the
floor area by use around each node and the walk to the nearest transit entrance. Controls for the regression
(fit_footfall_regression):
NYCWalks makes its trips at shops, jobs and subway entrances, so without them the VLM's measures stand in for land use.
Each source a table of points in data/<city>/public_data/downloads/:
  - lots: tax lots with floor areas by use (Murray Hill: NYC MapPLUTO); within radius_m of a node:
      retail_share, office_share, residential_share  of the floor area
      land_use_mix   the normalised entropy of retail / office / residential / other floor area (0 = one use)
      floor_area     the total floor area
  - transit: entrances (Murray Hill: MTA subway entrances); subway_distance_m to the nearest one (straight line)
  - seating: places to sit set out on the street (Murray Hill: the city's public seating, and the sidewalk cafes
      approved under Open Restaurants), each counted within seating.radius_m: seating_<source>, seating_formal

Writes data/<city>/public_data/land-use-per-node.csv. Run by pipeline step measure_city_records; read
through the node table by validate_vlm_ratings (the seating checks the VLM's seating ratings), compose_metric (seating
in M)
and fit_footfall_regression (controls).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point

from common.data_readers import record_map_layer
from common.downloads import record, socrata_points, socrata_shapes, socrata_url
from common.node_table import node_table
from public_data.new_york.measure_street_canyons import FOOTPRINTS

# ---------------------------------------------------------------- 1. sources
# Open-data (Socrata) datasets on NYC Open Data (data.cityofnewyork.us; the MTA's on data.ny.gov), free to use under
# their terms. Each is downloaded the first time this step runs, only its rows around the nodes, and logged.
PORTAL = "data.cityofnewyork.us"
SOURCES = {
    "tax_lots": {                       # MapPLUTO: floor area by use, square feet
        "dataset": "64uk-42ks", "file": "mappluto-lots.csv", "lat_field": "latitude", "lon_field": "longitude",
        "fields": {"total": "bldgarea", "retail": "retailarea", "office": "officearea", "residential": "resarea"}},
    "subway_entrances": {               # MTA subway entrances and exits
        "portal": "data.ny.gov", "dataset": "i9wp-a4ja", "file": "subway-entrances.csv",
        "lat_field": "entrance_latitude", "lon_field": "entrance_longitude"},
    "public_seating": {                 # NYC DOT city benches and seating
        "dataset": "esmy-s8q5", "file": "nyc-public-seating.csv", "lat_field": "latitude", "lon_field": "longitude"},
    "sidewalk_cafes": {                 # Open Restaurants applications approved for sidewalk seating, 2020 on
        "dataset": "pitm-atqc", "file": "nyc-sidewalk-cafes.csv", "lat_field": "latitude", "lon_field": "longitude",
        "where": "approved_for_sidewalk_seating='yes'"},
    "sidewalk_shed_permits": {          # DOB NOW: Build approved permits; active when the photo was taken
        "dataset": "rbx6-tga4", "file": "dob-sidewalk-shed-permits.csv", "lat_field": "latitude",
        "lon_field": "longitude", "where": "work_type='Sidewalk Shed'",
        "fields": {"bin": "bin", "issued": "issued_date", "expired": "expired_date"}},
    "landmark_buildings": {             # Landmarks Preservation Commission designated buildings
        "dataset": "ncre-qhxs", "file": "landmark-buildings.csv", "lat_field": "latitude", "lon_field": "longitude",
        "fields": {"bin": "bin_number"}},
    "planimetric_sidewalks": {          # NYC planimetric sidewalks
        "dataset": "52n9-sdep", "file": "nyc-planimetric-sidewalks.geojson", "geometry_field": "the_geom"},
    "street_trees": {                   # NYC Parks Forestry Tree Points, living trees
        "dataset": "hn5i-inap", "file": "nyc-street-trees.geojson", "geometry_field": "location",
        "where": "tpstructure='Full'"},
    "storefront_filings": {             # Local Law 157 filings, the year closest to the April 2026 photos
        "dataset": "92iy-9c3n", "file": "nyc-storefront-filings.csv", "lat_field": "latitude",
        "lon_field": "longitude", "where": "reporting_year='2025'",
        "fields": {"vacant": "vacant_on_12_31", "bbl": "borough_block_lot"}},
    "privately_owned_public_spaces": {  # NYC privately owned public spaces
        "dataset": "rvih-nhyn", "file": "nyc-privately-owned-public-spaces.csv", "lat_field": "latitude",
        "lon_field": "longitude", "fields": {"borocode": "borocode", "block": "tax_block", "lot": "tax_lot"}},
    "temporary_public_art": {           # NYC DOT temporary art
        "dataset": "3r2x-bnmj", "file": "nyc-dot-temporary-art.csv", "lat_field": "latitude",
        "lon_field": "longitude", "fields": {"installed": "installation", "removed": "removal"}},
    "parks": {                          # NYC Parks properties
        "dataset": "enfh-gkve", "file": "nyc-parks-properties.geojson", "geometry_field": "multipolygon"},
    "pedestrian_plazas": {              # NYC DOT pedestrian plazas
        "dataset": "k5k6-6jex", "file": "nyc-dot-pedestrian-plazas.geojson", "geometry_field": "the_geom"},
}
LAND_USE = {"lots": "tax_lots", "transit": "subway_entrances"}
SEATING = {"public": "public_seating", "cafes": "sidewalk_cafes"}    # seating_<key> per node


def fetch(name, paths):
    """!
    @brief A source's file, downloaded around the nodes the first time and logged.
    """
    S, path = SOURCES[name], paths.public_data_downloads / SOURCES[name]["file"]
    url = socrata_url(S.get("portal", PORTAL), S["dataset"])
    if not path.exists():
        nodes = gpd.read_file(paths.nodes_gpkg)
        if "geometry_field" in S:
            socrata_shapes([url], S, nodes, path)
        else:
            socrata_points(url, S, nodes, path)
    record(paths, name, path, url)
    return path


def source(name, paths, crs):
    """!
    @brief One source's rows: shapes (geometry_field) or points (lat_field, lon_field).
    """
    S, path = SOURCES[name], fetch(name, paths)
    if "geometry_field" in S:
        g = gpd.read_file(path).to_crs(crs)
        return g.set_geometry(g.geometry.make_valid())     # a few city polygons self-touch, which intersection rejects
    return points(S, path, crs)


def points(S, path, crs):
    """!
    @brief The source's rows as points in the projected CRS, its `fields` renamed to their keys.
    """
    d = pd.read_csv(path).dropna(subset=[S["lat_field"], S["lon_field"]])
    d = d.rename(columns={v: k for k, v in S.get("fields", {}).items()})
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d[S["lon_field"]], d[S["lat_field"]]), crs=4326).to_crs(crs)


def table(name, paths):
    """!
    @brief A source's rows as read, every column text, its `fields` renamed to their keys (for records matched by
           building number, not by place).
    """
    return pd.read_csv(fetch(name, paths), dtype=str).rename(columns={v: k for k, v in SOURCES[name]["fields"].items()})


# ---------------------------------------------------------------- 2. frontage geometry
def frontage(nodes, sidewalks, buildings, R):
    """!
    @brief The sidewalk, the ground to the facade and the street wall per node, from rays square to the street.
    @param nodes      GeoDataFrame node_id, street_axis_deg (the street's bearing), geometry; metric CRS.
    @param sidewalks  Sidewalk polygons; buildings: footprint polygons; same CRS.
    @param R          The config's `rays`: reach_m, offsets_m (along the street), wall_m, max_sidewalk_m.
    @return DataFrame node_id, sidewalk_width_m, curb_to_facade_m, setback_m, street_wall_share.
    """
    side_tree, side_geoms = STRtree(sidewalks.geometry.values), sidewalks.geometry.values
    wall_tree, wall_geoms = STRtree(buildings.geometry.values), buildings.geometry.values

    def first_hit(ray, origin):
        d = [origin.distance(ray.intersection(wall_geoms[i])) for i in wall_tree.query(ray)
             if not ray.intersection(wall_geoms[i]).is_empty]
        return min(d) if d else np.nan

    def first_sidewalk(ray, origin):
        """the sidewalk the ray meets first: (distance to it, its width along the ray)"""
        segs = sorted((origin.distance(g), g.length) for i in side_tree.query(ray)
                      for g in getattr(ray.intersection(side_geoms[i]), "geoms", [ray.intersection(side_geoms[i])])
                      if not g.is_empty and g.length > 0)
        return segs[0] if segs else (np.nan, np.nan)

    rows = []
    for r in nodes.itertuples():
        if not np.isfinite(r.street_axis_deg):
            rows.append({"node_id": r.node_id})
            continue
        a = np.deg2rad(r.street_axis_deg)
        along, across = np.array([np.sin(a), np.cos(a)]), np.array([np.cos(a), -np.sin(a)])
        sides = []
        for s in (1, -1):
            curb, width, wall = [], [], []
            for o in R["offsets_m"]:
                p0 = np.array([r.geometry.x, r.geometry.y]) + along * o
                origin, ray = Point(p0), LineString([p0, p0 + s * across * R["reach_m"]])
                c, w = first_sidewalk(ray, origin)
                curb.append(c)
                width.append(w if w <= R["max_sidewalk_m"] else np.nan)    # longer: a ray running along a corner
                wall.append(first_hit(ray, origin))
            curb, width, wall = map(np.array, (curb, width, wall))
            ground = wall - curb
            sides.append({"width": np.nanmedian(width) if np.isfinite(width).any() else np.nan,
                          "ground": np.nanmedian(ground) if np.isfinite(ground).any() else np.nan,
                          "wall": np.mean(np.nan_to_num(ground, nan=np.inf) <= np.nan_to_num(width, nan=0) + R["wall_m"])})
        S = pd.DataFrame(sides)
        rows.append({"node_id": r.node_id, "sidewalk_width_m": S.width.mean(), "curb_to_facade_m": S.ground.mean(),
                     "setback_m": (S.ground - S.width).clip(lower=0).max(), "street_wall_share": S.wall.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. the buildings fronting a node
def active_on(permits, day):
    """!
    @brief The records running on `day` (issued <= day <= expired; no expiry = still running).
    """
    return permits[(permits.issued <= day) & (permits.expired.isna() | (permits.expired >= day))]


def records_per_node(nodes, buildings, permits, photo_month, landmark_bins, margin_m):
    """!
    @brief What is recorded for the buildings fronting each node: those within half the street's width plus margin_m.
    @param nodes        GeoDataFrame node_id, W_facade (metres, may be NaN), geometry, in a metric CRS.
    @param buildings    GeoDataFrame bin, year, geometry, in the same CRS.
    @param permits      DataFrame bin, issued, expired (datetimes).
    @param photo_month  {node_id: "YYYY-MM"} of its panorama; a node without one has no shed value.
    @param landmark_bins  The building numbers of designated landmarks.
    @return DataFrame node_id, buildings_fronting, median_building_year, landmark_building_share, active_sidewalk_shed.
    """
    rows = []
    for r in nodes.itertuples():
        reach = (r.W_facade / 2 if pd.notna(r.W_facade) else margin_m) + margin_m
        b = buildings[buildings.distance(r.geometry) <= reach]
        month = photo_month.get(r.node_id)
        shed = np.nan if pd.isna(month) else float(active_on(permits, pd.Timestamp(f"{month}-15")).bin.isin(b.bin).any())
        rows.append({"node_id": r.node_id, "buildings_fronting": len(b), "median_building_year": b.year.median(),
                     "landmark_building_share": b.bin.isin(landmark_bins).mean() if len(b) else np.nan,
                     "active_sidewalk_shed": shed})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. within the node's reach
def points_within(nodes, pts, radius_m):
    """!
    @brief How many of the points lie within radius_m of each node.
    """
    if not len(pts):
        return np.zeros(len(nodes), int)
    near = cKDTree(np.c_[pts.geometry.x, pts.geometry.y]).query_ball_point(np.c_[nodes.geometry.x, nodes.geometry.y],
                                                                             radius_m)
    return np.array([len(i) for i in near])


def shapes_within(nodes, shapes, radius_m):
    """!
    @brief How many of the shapes (footprints) come within radius_m of each node.
    """
    hit = shapes.sindex.query(nodes.geometry.buffer(radius_m).values, predicate="intersects")[0]
    return np.bincount(hit, minlength=len(nodes))


def storefronts(nodes, filings, buildings, radius_m):
    """!
    @brief The storefronts filed for the buildings within radius_m of each node (matched by tax lot: a filing's point
           is its lot's, often farther from the street than its shopfront), and the share of them filed as vacant.
    @param buildings  Footprints with bbl, in the nodes' CRS.
    @return (count, vacant share; NaN where there is none) per node.
    """
    f = filings.assign(bbl=pd.to_numeric(filings.bbl, errors="coerce").astype("Int64").astype(str),
                       vacant=filings.vacant.astype(str).str.upper().str.startswith("Y"))
    per_lot = f.groupby("bbl").agg(n=("vacant", "size"), v=("vacant", "sum"))
    b = buildings[buildings.bbl.isin(per_lot.index)].drop_duplicates("bbl")
    node_i, b_i = b.sindex.query(nodes.geometry.buffer(radius_m).values, predicate="intersects")
    hit = pd.DataFrame({"node": node_i, "bbl": b.bbl.to_numpy()[b_i]}).drop_duplicates().join(per_lot, on="bbl")
    s = hit.groupby("node")[["n", "v"]].sum().reindex(range(len(nodes)), fill_value=0)
    return s.n.to_numpy(), np.where(s.n > 0, s.v / s.n.where(s.n > 0, 1), np.nan)


def in_place(art, photo_month):
    """!
    @brief For each node, the art in place in its photo's month (installed <= the 15th <= removed).
    @return {node_id: GeoDataFrame}.
    """
    art = art.assign(installed=pd.to_datetime(art.installed, errors="coerce"),
                     expired=pd.to_datetime(art.removed, errors="coerce"), issued=lambda t: t.installed)
    return {n: active_on(art, pd.Timestamp(f"{m}-15")) for n, m in photo_month.items() if pd.notna(m)}


# ---------------------------------------------------------------- 5. land use and seating
USES = ["retail", "office", "residential"]


def mix(area):
    """!
    @brief Normalised entropy of the floor-area shares in each row (0 = one use, 1 = an even mix).
    """
    p = area / area.sum(axis=1, keepdims=True).clip(min=1e-9)
    h = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(axis=1)
    return h / np.log(area.shape[1])


def land_use(nodes, lots, transit, radius_m):
    """!
    @brief Per node: floor-area shares by use and their mix within radius_m, total floor area, and the distance
           to the nearest transit entrance.
    @param lots     Points with total, retail, office, residential floor area (or None).
    @param transit  Points (or None).
    """
    xy = np.c_[nodes.geometry.x, nodes.geometry.y]
    out = pd.DataFrame({"node_id": nodes.node_id.to_numpy()})
    if lots is not None:
        a = lots[["total", *USES]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        near = cKDTree(np.c_[lots.geometry.x, lots.geometry.y]).query_ball_point(xy, radius_m)
        s = np.array([a[i].sum(0) if i else np.zeros(a.shape[1]) for i in near])
        total = s[:, 0]
        for j, u in enumerate(USES, 1):
            out[f"{u}_share"] = np.where(total > 0, s[:, j] / np.maximum(total, 1), np.nan)
        other = np.clip(total - s[:, 1:].sum(1), 0, None)
        out["land_use_mix"] = np.where(total > 0, mix(np.c_[s[:, 1:], other]), np.nan)
        out["floor_area"] = total
    if transit is not None:
        out["subway_distance_m"] = cKDTree(np.c_[transit.geometry.x, transit.geometry.y]).query(xy)[0]
    return out


def seating(nodes, sources, radius_m):
    """!
    @brief Per node: the places to sit of each source within radius_m (seating_<source>), and their sum
           (seating_formal). Formal seating, set out by the city or a business: benches, cafe tables; the VLM's
           seating term counts the informal kind (stoops, ledges), so the two complement each other.
    @param sources  {name: points}.
    """
    xy = np.c_[nodes.geometry.x, nodes.geometry.y]
    out = pd.DataFrame({"node_id": nodes.node_id.to_numpy()})
    for name, p in sources.items():
        out[f"seating_{name}"] = [len(i) for i in cKDTree(np.c_[p.geometry.x, p.geometry.y]).query_ball_point(xy, radius_m)]
    out["seating_formal"] = out[[f"seating_{n}" for n in sources]].sum(axis=1)
    return out


def land_use_per_node(cfg, paths):
    """!
    @brief land-use-per-node.csv from the LAND_USE and SEATING sources.
    """
    P = cfg["city_records_per_node"]
    crs = int(cfg["projected_crs"])
    nodes = gpd.read_file(paths.nodes_gpkg).to_crs(crs)
    src = {key: source(name, paths, crs) for key, name in LAND_USE.items()}
    out = land_use(nodes, src["lots"], src["transit"], P["land_use"]["radius_m"])
    pts = {key: source(name, paths, crs) for key, name in SEATING.items()}
    out = out.merge(seating(nodes, pts, P["seating"]["radius_m"]), on="node_id")
    layer = lambda name: {**{k: SOURCES[name][k] for k in ("lat_field", "lon_field", "fields") if k in SOURCES[name]},
                          "file": str((paths.public_data_downloads / SOURCES[name]["file"]).relative_to(paths.city_data))}
    for key, name in LAND_USE.items():                            # the land-use and seating maps (make_figures)
        record_map_layer(paths, key, layer(name))
    record_map_layer(paths, "seating", {key: layer(name) for key, name in SEATING.items()})
    out.to_csv(paths.land_use_csv, index=False)
    print("land use per node:\n" + out.drop(columns="node_id").describe().loc[["mean", "50%", "max"]].round(3).to_string())


# ---------------------------------------------------------------- 6. run
def run(cfg, paths):
    """!
    @brief land-use-per-node.csv (land_use_per_node), then city-records-per-node.csv.
    """
    land_use_per_node(cfg, paths)
    P = cfg["city_records_per_node"]
    crs = int(cfg["projected_crs"])
    nodes = node_table(cfg, paths)
    if "street_axis_deg" not in nodes or "W_facade" not in nodes:
        raise SystemExit("needs the view indices (calculate_segmentation_pixel_shares) and the street canyons (measure_street_canyons)")
    # nodes without view indices (no usable photo) take their walking heading as the street's bearing
    nodes = nodes.reset_index(drop=True).assign(street_axis_deg=lambda t: t.street_axis_deg.fillna(t.heading_fwd_deg))
    src = {name: source(name, paths, crs) for name in ("planimetric_sidewalks", "street_trees", "storefront_filings",
           "privately_owned_public_spaces", "temporary_public_art", "parks", "pedestrian_plazas")}

    F = FOOTPRINTS                                                # downloaded by the street canyons (measure_street_canyons)
    b = gpd.read_file(paths.public_data_downloads / F["file"]).to_crs(crs)
    b = b.assign(bin=b[F["bin_field"]].astype(str),
                 year=pd.to_numeric(b[F["year_field"]], errors="coerce").replace(0, np.nan),
                 bbl=pd.to_numeric(b[F["bbl_field"]], errors="coerce").astype("Int64").astype(str))
    meta = pd.read_csv(paths.pano_metadata_csv)
    photo_month = dict(zip(meta.node_id, meta.pano_date))
    permits = table("sidewalk_shed_permits", paths).assign(issued=lambda t: pd.to_datetime(t.issued, errors="coerce"),
                                                           expired=lambda t: pd.to_datetime(t.expired, errors="coerce")
                                                           ).dropna(subset=["issued", "expired"])
    landmark_bins = set(table("landmark_buildings", paths).bin)

    out = frontage(nodes, src["planimetric_sidewalks"], b, P["rays"])
    out = out.merge(records_per_node(nodes, b[["bin", "year", "geometry"]], permits, photo_month, landmark_bins,
                                     P["fronting_margin_m"]), on="node_id")
    r = P["radius_m"]
    out["street_tree_count"] = points_within(nodes, src["street_trees"], r)
    out["storefronts"], out["vacant_storefront_share"] = storefronts(nodes, src["storefront_filings"], b, r)
    pops = src["privately_owned_public_spaces"]
    pops_bbl = set((pd.to_numeric(pops.borocode) * 10 ** 9 + pd.to_numeric(pops.block) * 10 ** 4
                    + pd.to_numeric(pops.lot)).astype("Int64").astype(str))
    out["pops_buildings"] = shapes_within(nodes, b[b.bbl.isin(pops_bbl)], r)
    out["landmark_buildings"] = shapes_within(nodes, b[b.bin.isin(landmark_bins)], r)
    art = in_place(src["temporary_public_art"], photo_month)
    out["public_art"] = [points_within(nodes.iloc[[i]], art.get(n, src["temporary_public_art"].iloc[:0]), r)[0]
                         for i, n in enumerate(nodes.node_id)]
    out["cultural_anchor_count"] = out.pops_buildings + out.landmark_buildings + out.public_art
    open_space = pd.concat([src["parks"].geometry, src["pedestrian_plazas"].geometry]).union_all()
    out["open_space_distance_m"] = nodes.geometry.distance(open_space).to_numpy()

    out.to_csv(paths.city_records_csv, index=False)
    use = out[out.node_id.isin(nodes.node_id[nodes.usable])]
    print(f"physical attributes, {len(use)} usable nodes:\n"
          + use.drop(columns="node_id").describe(percentiles=[.1, .5, .9]).T[["count", "mean", "10%", "50%", "90%", "max"]]
          .round(2).to_string())
