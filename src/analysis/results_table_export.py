"""!
@package results_table_export
@brief Every result per node in one table (GeoJSON for QGIS, and CSV).

One row per sampling node, usable or not:
  - the node: node_id, street, osm_name, typology, usable, exclude_reason, lat, lon
  - its measurements (node_table): view indices (GVI, VEI, ...), block face, canyon (H_m, W_facade, H/W, SVF_canyon),
    Space Syntax (choice, integration), land use, physical attributes (step measure_city_records)
  - its views' scores (step compose_metric), averaged over the node's views: the terms, the indicators, the dimensions
  and M;
    n_views
  - the outcome (step measure_pedestrian_volume): volume, each period's modelled and observed columns
  - street_type (step cluster_street_types; the views' most common type), dwell_potential and lingering at each s (step
  map_dwell_potential)

Writes outputs/<run_id>/: all-results-per-node.geojson (EPSG:4326) and all-results-per-node.csv. Run by
pipeline step export_all_results.
"""
import pandas as pd

from common.data_readers import read_outcome
from common.node_table import node_table


def all_results(cfg, paths):
    """!
    @brief The per-node table of every result.
    @return GeoDataFrame in EPSG:4326, one row per sampling node.
    """
    t = node_table(cfg, paths).to_crs(4326)
    t = t.assign(lat=t.geometry.y, lon=t.geometry.x)
    s = pd.read_csv(paths.scores_csv)
    scores = [c for c in s.select_dtypes("number") if c not in t and c != "seq"]   # terms, indicators, dimensions, M
    means = s.groupby("node_id")[scores].mean().join(s.groupby("node_id").size().rename("n_views")).reset_index()
    t = t.merge(means, on="node_id", how="left")
    t = t.merge(read_outcome(paths), on="node_id", how="left")
    m = pd.read_csv(paths.modelled_volume_csv).drop(columns="value")
    t = t.merge(m.rename(columns={"n_segments": "volume_segments"}), on="node_id", how="left")
    c = pd.read_csv(paths.clusters_csv).groupby("node_id").cluster.agg(lambda x: x.mode().iloc[0])
    t = t.merge(c.rename("street_type").reset_index(), on="node_id", how="left")
    t["street_type"] = t.street_type.astype("Int64")       # unusable nodes have none
    dp = pd.read_csv(paths.dwell_potential_nodes_csv)
    dp = dp[["node_id", "dwell_potential", *[c for c in dp if c.startswith("lingering_s")]]]
    return t.merge(dp, on="node_id", how="left")


# column -> what it is, for the README; the rated terms, indicators and pillars are described from the config
COLUMNS = {
    "the node": {
        "node_id": "the node's id: `<street folder>_<number>` (join key for every other table of the run)",
        "street / osm_name": "street folder name / OpenStreetMap street name",
        "lat / lon": "node position, WGS84 degrees",
        "heading_fwd_deg / heading_rev_deg": "the two walking directions, degrees from north; the two views look along these",
        "seq_fwd / seq_rev / chain_pos_m": "order along the street and metres from its start",
        "typology": "street group by name ({typology}, other): the city's sample_osm_nodes.typology, first match wins",
        "usable / exclude_reason": "False = excluded in review or no usable photo: such a node has no ratings or scores",
        "street_axis_deg": "the street's bearing, degrees",
        "face_id": "block-face chain; nodes on one chain are not independent (cluster errors by it)",
        "snap_m": "distance from the node to the network segment its Space Syntax values come from, m",
    },
    "measured from the photos' segmentation (step calculate_segmentation_pixel_shares)": {
        "GVI": "green view: segmented vegetation, % of the 360-degree view",
        "GVI_eye_seg": "segmented vegetation in the eye-level band (0 to 15 degrees below the horizon), %",
        "SVF_band / BVF_band / PVI_band": "share of sky / buildings / pavement in the level photos' band (not a true sky view factor)",
        "VEI": "visual enclosure index from the segmentation",
    },
    "street canyon from building footprints (step measure_street_canyons)": {
        "H_m": "median height of the buildings within 35 m, m",
        "W_facade / HW_facade": "facade-to-facade street width, m / height to width ratio",
        "HW_effective / HW_source": "H/W including values inherited at corners, and how it was obtained",
        "SVF_canyon": "sky view factor of the canyon floor, cos(arctan(2 H/W)), 0-1",
        "facade_sides_hit / facade_sides_open / facade_empty_bearing": "bookkeeping of the width probe",
    },
    "street network, Space Syntax (step measure_space_syntax)": {
        "choice / integration": "angular choice and integration within 800 m along the network",
        "choice_400 / integration_400 / choice_1600 / integration_1600": "the same within 400 m and 1,600 m",
    },
    "land use and seating (step measure_city_records)": {
        "retail_share / office_share / residential_share": "share of floor area within 75 m by use (MapPLUTO)",
        "land_use_mix / floor_area": "normalised entropy of the uses, 0-1 / total floor area within 75 m, sq ft",
        "subway_distance_m": "straight-line distance to the nearest subway entrance, m",
        "seating_public / seating_cafes / seating_formal": "city benches / approved sidewalk cafes / both, within 20 m",
    },
    "physical attributes from city records and footprints (step measure_city_records)": {
        "sidewalk_width_m / curb_to_facade_m / setback_m": "sidewalk width, curb to first facade, setback beyond the sidewalk (deeper side), m; rays square to the street",
        "street_wall_share": "share of rays meeting a facade within 10 ft of the sidewalk, 0-1",
        "buildings_fronting / median_building_year / landmark_building_share": "buildings fronting the node, their median construction year, share that are designated landmarks",
        "active_sidewalk_shed": "1 if a sidewalk-shed permit was active on those buildings in the photo's month",
        "street_tree_count": "living street trees within 20 m",
        "storefronts / vacant_storefront_share": "storefront filings (Local Law 157, 2025) for buildings within 20 m, and the share filed vacant (empty: no storefront)",
        "pops_buildings / landmark_buildings / public_art / cultural_anchor_count": "POPS and landmark buildings within 20 m (by footprint), temporary art in place in the photo's month, and their sum",
        "open_space_distance_m": "distance to the nearest park or DOT pedestrian plaza, m",
    },
    "outcome and predictions (steps measure_pedestrian_volume, cluster_street_types and map_dwell_potential)": {
        "n_views": "views averaged into the node's scores (2 for a usable node)",
        "volume": "NYCWalks modelled pedestrians per hour, weekday mean of the AM, midday and PM peaks, segments within 25 m: modelled, not observed",
        "volume_segments": "NYCWalks segments within 25 m",
        "predwkdyAM / predwkdyMD / predwkdyPM": "the three peak-hour estimates",
        "observed_predwkdyAM / observed_predwkdyMD / observed_predwkdyPM": "NYC DOT counts where a counted segment is near (21 nodes)",
        "street_type": "k-means street type of the node's views (1-3, step cluster_street_types)",
        "dwell_potential": "predicted lingering, volume x M / 7: a prediction, no lingering data exists",
        "lingering_s0.5 / lingering_s1 / lingering_s2": "volume x (M / 7)^s at s = 0.5, 1, 2",
    },
}
# earlier terms are stored under the paper's names
PAPER_NAMES = {"v_nat": "V_nat", "v_built": "V_built", "gvi_eye": "GVI_eye", "gmi": "GMI", "v_sign": "V_sign",
               "v_pave": "V_pave", "gfapi": "GFAPI", "ias": "IAS"}


def readme(cfg, t):
    """!
    @brief all-results-per-node.README.md: what the table is, how to load it, and every column.
    @param t  all_results().
    """
    from common.data_readers import dimension_columns, indicator_ratings
    C, terms = cfg["metric"], cfg["vlm"]["prompts"]["terms"]
    dims = list(C["dimensions"])
    usable = t[t.usable.astype(bool)]
    first = lambda s: " ".join(s.split()).split(": ", 1)[-1].split(". ")[0].rstrip(".")
    L = [f"# all-results-per-node.csv: {cfg['name']}, run `{cfg['run_id']}`", "",
         f"One row per sampling node: {len(t)} nodes, {len(usable)} usable, {t.shape[1] - 1} columns. Written by step export_all_results "
         "from the run's tables; `all-results-per-node.geojson` is the same table "
         "as points for QGIS. This file is written with it.", "",
         "## What the study does", "",
         "Nodes every 20 m along the study streets; two 180-degree Street View views per node. A vision-language model "
         f"(`{cfg['vlm']['model']}`) rates {len(terms)} terms on each view (1-7, the expected rung over its rung "
         "probabilities); step measure_city_records measures physical attributes from NYC records and building footprints. Indicators "
         "put ratings and records on 0-1 between fixed goalposts; the pillars average their indicators; "
         f"M = ({' x '.join(C['names'][k] for k in dims)})^(1/3), the Street Interface Matrix score, on 1-7.", "",
         "## Loading", "", "```python", "import pandas as pd",
         f"d = pd.read_csv('all-results-per-node.csv')", "d = d[d.usable]          # scores exist for usable nodes only",
         "d.sort_values('M', ascending=False)[['node_id', 'osm_name', 'M', " + ", ".join(f"'{k}'" for k in dims) + "]].head()",
         "```", "",
         "- Scores are node means of the node's two views. The per-view ratings on 1-7, with each rung's probability, "
         "are in `street-interface-matrix/all-rated-views-with-inputs.csv` (join on `node_id`).",
         "- Nodes are spatially dependent: neighbours 20 m apart share most of what they see. Cluster by `face_id` or use "
         "spatial models; do not treat rows as independent.",
         "- `volume` is modelled (NYCWalks, Sevtsuk et al. 2025), not counted; its source states no licence: cite it, do not "
         "redistribute it. `dwell_potential` and `lingering_*` are predictions: no lingering data exists for these streets.",
         "- Missing values: records can be empty (no building, no storefront); indicators fill a missing input with its "
         "street's mean, so `index_*`, the pillars and M are complete for usable nodes.", ""]
    L += ["## M, the pillars and the indicators", "",
          "| column | meaning | scale |", "|---|---|---|",
          "| `M` | Street Interface Matrix score: geometric mean of the pillars | 1-7 |"]
    L += [f"| `{k}` / `{k}_raw` | {C['names'][k]} pillar: 1 + 6 x the mean of its indicators / that mean | 1-7 / 0-1 |"
          for k in dims]
    L += ["", "Each indicator (`index_<name>`, 0-1) averages its ratings, each scaled from rung 1 (0) to rung 7 (1), and "
          "takes half from its record where it has one, scaled between goalposts [worst, best] (log: on ln(1 + x)):", "",
          "| pillar | column | ratings | record, goalposts worst -> best |", "|---|---|---|---|"]
    for k, items in C["dimensions"].items():
        for col, it in zip(dimension_columns(C)[k], items):
            rec = (f"`{it['record']}` {it['goalposts'][0]} -> {it['goalposts'][1]}" + (" (log)" if it.get("log") else "")
                   if "record" in it else "-")
            rs = ", ".join(f"`{PAPER_NAMES.get(r, r)}`" + (" (reversed)" if it.get("rating_goalposts") == [7, 1] else "")
                           for r in indicator_ratings(it))
            L.append(f"| {C['names'][k]} | `{col}` | {rs or '-'} | {rec} |")
    L += ["", "## The rated terms", "",
          "Each term's column is its rating averaged over the node's views, on 0-1: (rating - 1) / 6, so 0 = rung 1 and "
          "1 = rung 7. View: along = the 180-degree view, facade = the two frames square to the frontage.", "",
          "| column | view | what the model rates |", "|---|---|---|"]
    L += [f"| `{PAPER_NAMES.get(f, f)}` | {c['view']} | {first(c['definition'])} |" for f, c in terms.items()]
    described = {f"`{c}`" for c in ["M", *dims, *[f"{k}_raw" for k in dims], *[PAPER_NAMES.get(f, f) for f in terms],
                                     *[c for cs in dimension_columns(C).values() for c in cs]]}
    typology = ", ".join(cfg["sample_osm_nodes"].get("typology") or {})       # the city's street groups
    for group, cols in COLUMNS.items():
        L += ["", f"## {group[0].upper() + group[1:]}", "", "| column | meaning |", "|---|---|"]
        L += [f"| `{k.replace(' / ', '` / `')}` | {v.replace('{typology}', typology)} |" for k, v in cols.items()]
        described |= {f"`{c}`" for k in cols for c in k.split(" / ")}
    other = [c for c in t.columns if c != "geometry" and f"`{c}`" not in described]
    if other:
        L += ["", "## Not yet described", "", ", ".join(f"`{c}`" for c in other)]
    L += ["", f"Settings: `configs/experiments/{cfg['run_id']}.yaml`; prompts: `configs/inference/vlm_prompts.yaml`; method: `docs/pipeline.md`.", ""]
    return "\n".join(L)


def run(cfg, paths):
    """!
    @brief Write all-results-per-node.geojson, .csv and its README.
    """
    t = all_results(cfg, paths)
    paths.all_results_geojson.unlink(missing_ok=True)       # the GeoJSON driver will not overwrite
    t.to_file(paths.all_results_geojson, driver="GeoJSON")
    pd.DataFrame(t.drop(columns="geometry")).to_csv(paths.all_results_csv, index=False)
    paths.all_results_readme.write_text(readme(cfg, t))
    print(f"all-results-per-node: {len(t)} nodes x {t.shape[1] - 1} columns ({int(t.usable.sum())} usable) "
          f"-> {paths.all_results_geojson.name}, {paths.all_results_csv.name}, {paths.all_results_readme.name}")
