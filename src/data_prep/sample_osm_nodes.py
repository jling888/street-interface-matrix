"""!
@package sample_osm_nodes
@brief Sampling nodes: camera nodes every spacing_m along each named street.

Steps (one function each, in this order):
  1. read_study_area     the study polygon (a boundary file or [lat, lng] corners), in degrees and in metres
  2. fetch_osm_ways      OSM highway ways in the polygon's bounding box, as of `osm_date`
  3. keep_study_streets  only the ways that are study streets (road class, name, tags)
  4. join_street_lines   each street's ways joined into lines, cut to the polygon (plus a small edge tolerance),
                         pointed south to north
  5. place_nodes         a node every spacing_m along each line, with its heading and walk order both ways;
                         street_typology labels each node's street
  6. draw_nodes_map      map of the nodes, a check before any imagery is bought (common.map_style)
  7. run                 writes data/<city>/data_prep/: nodes.gpkg, nodes.csv, study-streets.gpkg and nodes-map.png

Run by pipeline step sample_street_nodes.
"""
import json
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString

from common.data_readers import study_area
from common.downloads import overpass
from common.map_style import draw_nodes_map
from common.node_table import node_review, read_nodes


# ---------------------------------------------------------------- 1. study area
def read_study_area(F, paths, crs):
    """!
    @brief The study area, as the config's `polygon` gives it (a file in data/<city>/data_prep/ or [lat, lng] corners).
    @return (polygon in EPSG:4326, for the download box; the same in the metre CRS, for clipping).
    """
    area_ll = study_area(F["polygon"], paths.data_prep)
    return area_ll, gpd.GeoSeries([area_ll], crs=4326).to_crs(crs).iloc[0]


# ---------------------------------------------------------------- 2. OSM download
def fetch_osm_ways(s, w, n, e, cfg, paths):
    """!
    @brief Every OSM highway way in a bounding box, as of `osm_date`, with its tags.
    @param s, w, n, e  The box: south, west, north, east (degrees).
    @return GeoDataFrame in EPSG:4326, one LineString per way, OSM tags as columns.
    """
    query = 'way(%.6f,%.6f,%.6f,%.6f)["highway"];out tags geom;' % (s, w, n, e)
    elements = json.loads(overpass(query, cfg, paths, paths.data_prep_osm_cache).read_text())["elements"]
    elements = [el for el in elements if len(el.get("geometry") or []) >= 2]

    tags = [el.get("tags", {}) for el in elements]
    lines = [LineString([(pt["lon"], pt["lat"]) for pt in el["geometry"]]) for el in elements]
    return gpd.GeoDataFrame(tags, geometry=lines, crs=4326)


# ---------------------------------------------------------------- 3. street filter
def keep_study_streets(ways, F):
    """!
    @brief Only the ways that are study streets: the road classes in highway_keep, named, on street level, no
           tunnels, bridges or ramps, and with a name matching name_include if set.

    The rules read OSM tags, which mean the same in every city; the city's own choices are its config's highway_keep
    and name_include. Dropping unnamed ways, tunnels and bridges suits Manhattan: where walkable streets are often
    unnamed in OSM, make those rules settings rather than lose the streets.

    @param ways  GeoDataFrame from fetch_osm_ways(), OSM tags as columns.
    @param F     The config's `sample_osm_nodes`.
    @return The kept rows, with `name` filled ("" for unnamed).
    """
    def osm_tag(key, missing=""):
        """One OSM tag as strings, `missing` where a way lacks it."""
        col = ways[key] if key in ways else pd.Series(None, index=ways.index, dtype=object)
        return col.fillna(missing).astype(str)

    road, name = osm_tag("highway"), osm_tag("name")
    drop = (
        ~road.isin(F["highway_keep"])                                           # wrong road class
        | road.str.endswith("_link")                                            # ramps
        | ((road == "footway") & osm_tag("footway").isin(["sidewalk", "crossing"]))  # a road's own sidewalk
        | ((road == "pedestrian") & (osm_tag("area") == "yes"))                 # a square, not a line
        | (osm_tag("indoor") == "yes")                                          # inside a building
        | (osm_tag("layer", "0") != "0")                                        # below or above street level
        | (name == "")                                                          # unnamed
        | (osm_tag("tunnel", "no") != "no")
        | (osm_tag("bridge", "no") != "no")
    )
    if F.get("name_include"):                                                   # the city's street list
        drop |= ~name.map(re.compile(F["name_include"]).search).astype(bool)
    return ways[~drop].assign(name=name)


# ---------------------------------------------------------------- 4. street lines
def join_street_lines(ways, clip, F):
    """!
    @brief Each street's ways joined end to end and cut to the study area.

    Joining stops at a fork, and a divided road's two carriageways are separate
    OSM ways, so one name can give several lines; so can a street that leaves
    the study area and comes back. Pieces shorter than min_corridor_m are dropped.

    @param ways  Kept ways in the projected CRS, from keep_study_streets().
    @param clip  The study area in the projected CRS, grown by polygon_buffer_m.
    @param F     The config's `sample_osm_nodes`.
    @return {street name: [LineString, ...]}, each line south to north, lines ordered west to east.
    """
    out = {}
    for name, group in ways.groupby("name", sort=True):
        merged = shapely.line_merge(shapely.MultiLineString(list(group.geometry)))
        pieces = []
        for line in shapely.get_parts(merged):
            for piece in shapely.get_parts(shapely.line_merge(line.intersection(clip))):
                if piece.geom_type != "LineString" or piece.length < F["min_corridor_m"]:
                    continue
                # point it north (east if exactly east-west), so walks start at the south end
                (x0, y0), (x1, y1) = piece.coords[0], piece.coords[-1]
                pieces.append(piece.reverse() if (y1, x1) < (y0, x0) else piece)
        out[name] = sorted(pieces, key=lambda l: (l.centroid.x, l.centroid.y))   # west first: stable _2, _3
    return out


# ---------------------------------------------------------------- 5. nodes
def street_id_from_name(name):
    """!
    @brief A street name as an id: 'East 34th Street' -> 'east_34th_street'.
    @param name  OSM street name.
    @return Lower-case name, runs of non-word characters replaced by one underscore.
    """
    return re.sub(r"_+", "_", re.sub(r"[^\w]+", "_", name.strip().lower())).strip("_")


def points_along_line(line, d):
    """!
    @brief Points at distances along a line, clamped to its ends.
    @param line  shapely LineString.
    @param d     Distances from the line's start (m); values outside [0, length] are clamped.
    @return Array of shapely Points.
    """
    return shapely.line_interpolate_point(line, np.clip(d, 0, line.length))


def place_nodes(lines, F, crs):
    """!
    @brief A node every spacing_m along each line, from its south end.
    @param lines  {street name: [LineString, ...]} from join_street_lines().
    @param F      The config's `sample_osm_nodes`.
    @param crs    The metre CRS the lines are in.
    @return GeoDataFrame in `crs`, one row per node: node_id, street, osm_name, lat, lon,
            heading_fwd_deg, heading_rev_deg, seq_fwd, seq_rev, chain_pos_m.
    """
    rows = []
    for name, pieces in lines.items():
        for k, line in enumerate(pieces):
            d = np.arange(0, line.length + 1e-9, F["spacing_m"])   # 0, 20, 40, ... m along the line
            rows.append(pd.DataFrame({
                "street": street_id_from_name(name) + (f"_{k + 1}" if k else ""),  # second line of a name: _2, ...
                "osm_name": name,
                "seq_fwd": np.arange(1, len(d) + 1),
                "chain_pos_m": d.round(2),
                "geometry": points_along_line(line, d),
                "back": points_along_line(line, d - 0.5),                    # half a metre either side,
                "ahead": points_along_line(line, d + 0.5),                   # for the heading
            }))
    df = pd.concat(rows, ignore_index=True)

    ll = {c: gpd.GeoSeries(df[c], crs=crs).to_crs(4326) for c in ("geometry", "back", "ahead")}
    # compass bearing back -> ahead, against true north (not the projection's grid north)
    dx = (ll["ahead"].x - ll["back"].x) * np.cos(np.radians(ll["geometry"].y))
    heading = np.degrees(np.arctan2(dx, ll["ahead"].y - ll["back"].y)) % 360

    df["node_id"] = df.street + "_" + df.seq_fwd.map("{:03d}".format)
    df["lat"], df["lon"] = ll["geometry"].y.round(6), ll["geometry"].x.round(6)
    df["heading_fwd_deg"] = heading.round(1)                        # walking south to north
    df["heading_rev_deg"] = ((heading + 180) % 360).round(1)       # walking back
    df["seq_rev"] = df.groupby("street").seq_fwd.transform("max") - df.seq_fwd + 1

    cols = ["node_id", "street", "osm_name", "lat", "lon", "heading_fwd_deg", "heading_rev_deg",
            "seq_fwd", "seq_rev", "chain_pos_m", "geometry"]
    return gpd.GeoDataFrame(df[cols], geometry="geometry", crs=crs)


def street_typology(names, pats):
    """!
    @brief Label streets by the city's `typology` regexes (first match wins).
    @param names  Street names.
    @param pats   sample_osm_nodes.typology: {label: regex}, or None.
    @return Array of typology labels, 'other' where no regex matches.
    """
    pats = pats or {}
    s = pd.Series(list(names)).astype(str)
    if not pats:
        return np.full(len(s), "other", dtype=object)
    return np.select([s.str.contains(p, case=False, na=False) for p in pats.values()],
                     list(pats), default="other")


# ---------------------------------------------------------------- 7. run
def run(cfg, paths):
    """!
    @brief Place the nodes, write nodes.gpkg / nodes.csv / study-streets.gpkg, draw the map.
    """
    F, crs = cfg["sample_osm_nodes"], int(cfg["projected_crs"])
    print(f"OSM as of {cfg.get('osm_date') or 'today (not pinned)'}")

    # 1. study area: lon/lat for the download box, metres for clipping
    area_ll, area = read_study_area(F, paths, crs)

    # 2. OSM ways in the polygon's bounding box
    w, s, e, n = area_ll.bounds
    ways = fetch_osm_ways(s, w, n, e, cfg, paths).to_crs(crs)

    # 3. only the study streets
    kept = keep_study_streets(ways, F)
    print(f"{len(ways)} OSM ways, {len(kept)} kept")

    # 4. each street as lines, cut to the area (grown by the edge tolerance)
    lines = join_street_lines(kept, area.buffer(F["polygon_buffer_m"]), F)

    # 5. a node every spacing_m, with its street type
    nodes = place_nodes(lines, F, crs)
    nodes["typology"] = street_typology(nodes.osm_name, F.get("typology"))
    if nodes.node_id.duplicated().any():
        sys.exit("duplicate node ids")

    # the nodes (read by every later step; none of them writes to them), and the study
    #    streets around them, which the maps draw as context
    # replaced, not written into: a GeoPackage takes a new layer per layer name, and readers get the first
    for p in (paths.nodes_gpkg, paths.streets_gpkg):
        p.unlink(missing_ok=True)
    nodes.to_file(paths.nodes_gpkg, driver="GPKG")
    streets = kept[["name", "geometry"]].clip(area.buffer(60))
    streets[~streets.is_empty].to_file(paths.streets_gpkg, driver="GPKG")
    nodes.drop(columns="geometry").to_csv(paths.nodes_csv, index=False)

    # hand-made exclusions (node-review-decisions.csv); read_nodes() stops on an id not in these nodes
    read_nodes(paths)
    review = node_review(paths)
    listed = set(review.node_id[review.decision == "exclude"])
    print(f"excluded by review: {len(listed)}")

    print(f"{len(nodes)} nodes on {nodes.street.nunique()} streets")
    print(nodes.typology.value_counts().to_string())

    # 6. the map: a check of the nodes before any imagery is bought
    draw_nodes_map(cfg, nodes, area_ll, streets, listed, paths.nodes_map)
