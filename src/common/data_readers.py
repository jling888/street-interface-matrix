"""!
@package data_readers
@brief Readers of the files one stage writes and later stages read: the stages share files, never each other's code.

  study_area                 the study polygon (data_prep's input)
  footprints                 the buildings with their roof height (public_data)
  photo, label_map_path,     where a photo's label map is, and the class ids in it (inference)
  class_taxonomy, IGNORE
  TermSharesByBearing        term-shares-by-bearing.npz (inference)
  FIELDS, rated_fields       the terms in a ratings table (inference)
  pedestrian_volume_segments the NYCWalks segments, and read_outcome, the volume per node (public_data)
  arc_shares                 the segmented term shares over each view's own arc (inference)
  map_layers                 the raw files a city's public data names for the maps: which file, which columns
  TERMS, dimension_columns,  the metric's contract: its config's `dimensions` (indicators, index_<name> columns) and
  indicator_ratings/_name    `names`, which the analysis reads for any metric system
"""
import json
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


def study_area(p, raw_dir):
    """!
    @brief The study area as one lon/lat polygon.
    @param p        `sample_osm_nodes.polygon`: a boundary file (GeoJSON, shapefile, GeoPackage; any CRS, all shapes
                    merged) or a list of [lat, lng] corners.
    @param raw_dir  The folder a file name is relative to (data/<city>/data_prep/).
    @return shapely Polygon (or MultiPolygon) in EPSG:4326.
    """
    if isinstance(p, str):
        return gpd.read_file(raw_dir / p).to_crs(4326).union_all()
    return Polygon([(lng, lat) for lat, lng in p])


def footprints(paths):
    """!
    @brief The buildings with their roof height, as the public data writes them for any city
           (buildings-with-height.gpkg: height_m, geometry, in the projected CRS), or None when it has none.
    """
    return gpd.read_file(paths.buildings_with_height) if paths.buildings_with_height.exists() else None


IGNORE = 255                                                     # not counted anywhere (capture vehicle)


def photo(node_id, heading):
    """!
    @brief A photo's file stem, shared by its raw outputs and its label map.
    """
    return f"{node_id}_{int(heading):03d}"


def label_map_path(paths, node_id, heading):
    """!
    @brief Where the label map of one photo is saved.
    """
    return paths.fused_label_maps / f"{photo(node_id, heading)}.png"


def class_taxonomy(cfg):
    """!
    @brief The label maps' class ids: 0 unknown, then the classes of segmentation_classes.yaml in their priority order.
    """
    return {"unknown": 0, **{c: i + 1 for i, c in enumerate(cfg["classes"])}}


class TermSharesByBearing:
    """!
    @brief The layout of a node's term shares by bearing (term-shares-by-bearing.npz, one array per node): a row
           per term over the whole photo, then per term within the eye band (<term>_eye), then the two weight
           rows a term's bins are divided by to give a share; a column per bearing bin. validate_vlm_ratings and
           compose_metric read a view's own arc out of it.
    """

    def __init__(self, cfg):
        self.terms = list(cfg["terms"])
        self.rows = self.terms + [t + "_eye" for t in self.terms] + ["weight", "weight_eye"]
        self.nbins = cfg["segmentation_pixel_shares"]["bearing_bins"]

    def row(self, name):
        """!
        @brief Row of `name`: a term (segmentation_classes.yaml terms), <term>_eye, "weight" or "weight_eye".
        """
        return self.rows.index(name)

    def bin_mask(self, centre, fov):
        """!
        @brief The bearing bins within fov/2 of `centre` (degrees).
        @return Boolean array of nbins.
        """
        d = (np.arange(self.nbins) + 0.5 - centre) % 360
        d = np.where(d > 180, d - 360, d)
        return np.abs(d) <= fov / 2

    def shares(self, p, mask):
        """!
        @brief Every term's share of the view within the bearing bins `mask`, from one node's array `p`.
        @return {term: share, term_eye: share}; NaN where the bins hold no view.
        """
        row = self.row
        w, we = p[row("weight")][mask].sum(), p[row("weight_eye")][mask].sum()
        out = {t: p[row(t)][mask].sum() / w if w > 0 else np.nan for t in self.terms}
        out.update({f"{t}_eye": p[row(t + "_eye")][mask].sum() / we if we > 0 else np.nan for t in self.terms})
        return out

    def gvi_and_vei(self, p, centre, fov):
        """!
        @brief GVI (%) and VEI over the bearings within fov/2 of `centre`, from one node's array `p`.

        Dividing by the weight row makes the result a share regardless of how many photos cover a bin.
        """
        s = self.shares(p, self.bin_mask(centre, fov))
        veg, sky, bld = s["vegetation"], s["sky"], s["building"]
        return 100 * veg, (bld / (sky + bld) if (sky + bld) > 0 else np.nan)


# rated term -> the paper's term
FIELDS = {
    "v_nat": "V_nat",
    "gvi_eye": "GVI_eye",
    "gmi": "GMI",
    "v_built": "V_built",
    "enclosure": "1 - SVF",
    "v_sign": "V_sign",
    "v_pave": "V_pave",
    "gfapi": "GFAPI",
    "ias": "IAS",
}


def rated_fields(d):
    """!
    @brief The terms a ratings table holds, in its own order.
    """
    return [c[:-3] for c in d.columns if c.endswith("_ev")]


def pedestrian_volume_segments(path, nodes, margin_m):
    """!
    @brief The pedestrian volume's segments within margin_m of the nodes' extent, from a vector file or a zip
           holding one.
    @param nodes  GeoDataFrame in a metre CRS.
    """
    src = str(path)
    if path.suffix == ".zip":
        inner = next(n for n in zipfile.ZipFile(path).namelist()
                     if n.endswith((".shp", ".gpkg", ".geojson")) and not n.startswith("__MACOSX"))
        src = f"zip://{path}!{inner}"
    box = gpd.GeoSeries([nodes.union_all().envelope.buffer(margin_m)], crs=nodes.crs)
    return gpd.read_file(src, bbox=box).to_crs(nodes.crs)


def record_map_layer(paths, name, spec):
    """!
    @brief Name a downloaded file of the city's public data that the figures draw (map-layers.json), with how to
           read it: {"file": under data/<city>/, and its columns}.
    """
    layers = map_layers(paths)
    layers[name] = spec
    paths.map_layers_json.write_text(json.dumps(layers, indent=1))


def map_layers(paths):
    """!
    @brief {name: spec} of record_map_layer(), or {} before the public data has run.
    """
    return json.loads(paths.map_layers_json.read_text()) if paths.map_layers_json.exists() else {}


def read_outcome(paths):
    """!
    @brief The outcome per node, for the analysis.
    @return DataFrame node_id, volume (pedestrians per hour).
    """
    return pd.read_csv(paths.modelled_volume_csv).groupby("node_id", as_index=False).value.mean().rename(
        columns={"value": "volume"})


def arc_shares(d, node_shares, bearing, by_bearing, fov):
    """!
    @brief Each term's share over each view's own arc (`fov`, stitch_180_views.fov, centred on its bearing).
    @param d        Ratings rows (file, node_id).
    @param node_shares  {node_id: term shares by bearing} (term-shares-by-bearing.npz).
    @param bearing  views-180.csv bearing, indexed by file.
    @param by_bearing   TermSharesByBearing (the rows).
    @return DataFrame, one row per row of d: each term and <term>_eye, green_share(_eye), enclosure; NaN without
            measured shares.
    """
    rows = {}
    for i, r in d.iterrows():
        p, b = node_shares.get(r.node_id), bearing.get(r.file)
        if p is not None and pd.notna(b):
            rows[i] = by_bearing.shares(p, by_bearing.bin_mask(b, fov))
    out = pd.DataFrame(list(rows.values()), index=list(rows)).reindex(
        index=d.index, columns=[t + e for e in ("", "_eye") for t in by_bearing.terms])
    for e in ("", "_eye"):
        out["green_share" + e] = out["vegetation" + e] / (out["vegetation" + e] + out["building" + e])
    out["enclosure"] = 1 - out["sky"]
    return out


# rated term -> its column in the metric's tables, where the column name differs
TERMS = {"v_nat": "V_nat", "v_built": "V_built", "gvi_eye": "GVI_eye", "gmi": "GMI", "v_sign": "V_sign",
         "v_pave": "V_pave", "gfapi": "GFAPI", "ias": "IAS"}


def indicator_ratings(item):
    """!
    @brief The rated terms an indicator reads: `ratings`, or the one `rating`.
    """
    return item.get("ratings") or ([item["rating"]] if "rating" in item else [])


def indicator_name(item):
    """!
    @brief An indicator's column: index_<its name, else the rated term or the record>.
    """
    return "index_" + (item.get("name") or item.get("rating") or item["record"])


def dimension_columns(C):
    """!
    @brief {dimension: [its indicator columns]} from the metric config's `dimensions`.
    """
    return {k: [indicator_name(i) for i in items] for k, items in C["dimensions"].items()}
