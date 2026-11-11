"""!
@package calculate_segmentation_pixel_shares
@brief The fused label maps' pixel shares of each node's view, weighted by solid angle: each term's share at every
       bearing, the class shares per node, then GVI, VEI and the block faces. No model: arithmetic on the label maps.

Steps, in this order:
  1. shares        per node, from its label maps: each term's (segmentation_classes.yaml) share per degree of
                   bearing, over the whole photo and within the paper's foveal band (eye_band_deg), and the share of
                   each class in the node's view. The layout of the term shares by bearing, which steps compose_metric
                   and validate_vlm_ratings
                   read too, is common/data_readers.TermSharesByBearing
  2. view indices  the four views of the paper's terms, over the full 360: GVI (% vegetation; GVI_eye_seg
                   within the eye band), SVF_band / BVF_band / PVI_band (sky / building / sidewalk share
                   of the imaged +-45 degree band; SVF_band is NOT a sky view factor, the zenith is never
                   sampled; PVI_band is sidewalk as seen from the carriageway, where the camera is), and
                   VEI = building / (sky + building), the enclosure (trees do not count)
  3. block faces   the clustering unit, and Moran's I
  4. run           runs them (2 and 3 inline), plus a coverage audit

Runs on CPU in seconds, so changing the terms in segmentation_classes.yaml or the config's segmentation_pixel_shares
only needs this step.
Skipped without label maps. Writes data/<city>/inference/segmentation/<variant>/measurements/:
term-shares-by-bearing.npz
(one array per node, rows TermSharesByBearing.rows), class-shares-per-node.csv,
view-indices-per-node.csv (GVI_eye_seg: GVI within the foveal band), view-indices-per-block-face.csv.
Run by pipeline step calculate_segmentation_pixel_shares.
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

from common.data_readers import IGNORE, TermSharesByBearing, class_taxonomy, label_map_path
from common.node_table import read_nodes, street_axis
from inference.fuse_segmentation_labels import check_settings


# ---------------------------------------------------------------- 1. shares
def term_ids(cfg, tax):
    """!
    @brief Each term's class ids (the terms' classes are checked on load).
    @param tax  The taxonomy, {class: id} (class_taxonomy).
    @return [[id, ...], ...] in the order of segmentation_classes.yaml terms.
    """
    return [[tax[c] for c in classes] for classes in cfg["terms"].values()]


def column_bearings(heading, width, fov):
    """!
    @brief Bearing of each pixel column of a perspective (gnomonic) photo.
    @param fov  The photo's horizontal field of view (download_gsv_panos.fov).
    @return Degrees, one per column.
    """
    x = (np.arange(width) + 0.5) / width * 2 - 1
    return (heading + np.degrees(np.arctan(x * np.tan(np.radians(fov / 2))))) % 360


def column_weights(height, width, fov, solid_angle):
    """!
    @brief Solid angle per column, normalised to sum to 1 (the corners of a perspective photo subtend
           less than its centre).
    @param solid_angle  segmentation_pixel_shares.solid_angle_weighting; equal weights without it.
    @return One weight per column.
    """
    if not solid_angle:
        return np.full(width, 1.0 / width)
    t = np.tan(np.radians(fov / 2))
    x = ((np.arange(width) + 0.5) / width * 2 - 1) * t
    y = ((np.arange(height) + 0.5) / height * 2 - 1) * t
    X, Y = np.meshgrid(x, y)
    w = (1 + X**2 + Y**2) ** -1.5
    return w.sum(axis=0) / w.sum()


def eye_rows(height, width, fov, band):
    """!
    @brief Which pixels of a level perspective photo lie in `band` (eye_band_deg: elevation, degrees).
    @return HxW bool array.
    """
    t = np.tan(np.radians(fov / 2))
    x = ((np.arange(width) + 0.5) / width * 2 - 1) * t
    y = ((np.arange(height) + 0.5) / height * 2 - 1) * t                 # downwards
    X, Y = np.meshgrid(x, y)
    elev = np.degrees(-np.arctan(Y / np.sqrt(1 + X**2)))
    lo, hi = band
    return (elev >= lo) & (elev <= hi)


def node_term_shares(rows, cfg, paths, tax, by_bearing, ids):
    """!
    @brief A node's term shares by bearing and its class shares, from its saved label maps.

    Each photo column adds, at its bearing, the share of its pixels in each term (weighted by the
    column's solid angle), once over the whole column and once over its pixels in the eye band; the
    weight rows add the column's weight over non-ignored pixels (whole, in the band), so dividing a
    term's bins by them gives a share whatever the overlap between photos.

    @param rows  The node's rows of street-view-frames.csv.
    @param tax   The taxonomy, {class: id}.
    @param by_bearing  TermSharesByBearing (the rows).
    @param ids   term_ids().
    @return (array len(by_bearing.rows) x bearing_bins, {class: share of the node's view}); (None, None) if a
            label map is missing.
    """
    P, fov = cfg["segmentation_pixel_shares"], cfg["download_gsv_panos"]["fov"]
    n = len(ids)
    acc = np.zeros((len(by_bearing.rows), P["bearing_bins"]))
    weight = np.zeros(256)
    for r in rows.itertuples():
        path = label_map_path(paths, r.node_id, r.heading)
        if not path.exists():
            return None, None
        a = np.asarray(Image.open(path))
        H, W = a.shape
        cw = column_weights(H, W, fov, P["solid_angle_weighting"])
        eye = eye_rows(H, W, fov, P["eye_band_deg"])
        idx = np.floor(column_bearings(float(r.heading), W, fov)).astype(int) % P["bearing_bins"]
        for k, term in enumerate(ids):
            hit = np.isin(a, term)
            np.add.at(acc[k], idx, hit.mean(axis=0) * cw)
            np.add.at(acc[n + k], idx, (hit & eye).mean(axis=0) * cw)
        seen = a != IGNORE
        np.add.at(acc[by_bearing.row("weight")], idx, seen.mean(axis=0) * cw)
        np.add.at(acc[by_bearing.row("weight_eye")], idx, (seen & eye).mean(axis=0) * cw)
        weight += np.bincount(a.ravel(), weights=np.broadcast_to(cw / H, a.shape).ravel(), minlength=256)
    total = weight[:IGNORE].sum()
    return acc, {c: weight[i] / total if total else np.nan for c, i in tax.items()}


# ---------------------------------------------------------------- 3. block faces
def morans_i(vals, xy, radius, nperm, seed):
    """!
    @brief Moran's I with inverse-distance weights inside `radius`.
    @return (I, permutation p-value); NaN, NaN without neighbour pairs.
    """
    pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.nan, np.nan
    w = 1.0 / np.maximum(np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1), 1.0)
    z = vals - vals.mean()
    den, S0 = (z ** 2).sum(), w.sum() * 2
    stat = lambda zz: (len(vals) / S0) * ((w * zz[pairs[:, 0]] * zz[pairs[:, 1]]).sum() * 2 / den)
    I = stat(z)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(z)) for _ in range(nperm)])
    return I, (np.sum(np.abs(null) >= abs(I)) + 1) / (nperm + 1)


# ---------------------------------------------------------------- 4. run
def audit(paths, missing):
    """!
    @brief Print why each node has no metric: its exclude_reason, else it was never measured.
    """
    reason = read_nodes(paths).set_index("node_id").exclude_reason
    print(pd.Series([reason.get(n) or "has imagery but no complete label maps (rerun fuse_label_maps)"
                     for n in missing]).str.split(" [(]").str[0].value_counts().to_string())


def run(cfg, paths):
    """!
    @brief Term shares by bearing for every node whose label maps are all there, then the node metrics.
    """
    P = cfg["segmentation_pixel_shares"]
    tax, by_bearing = class_taxonomy(cfg), TermSharesByBearing(cfg)
    ids = term_ids(cfg, tax)

    # the fused label maps, made with the current settings
    maps = paths.fused_label_maps
    if not (maps.exists() and any(maps.iterdir())):
        print(f"no label maps in {maps}; run fuse_label_maps first. Skipping.")
        return
    check_settings(cfg, paths)         # label maps made with other class ids would be misread
    paths.label_map_measurements.mkdir(parents=True, exist_ok=True)

    # 1. the term shares by bearing of every node with every label map
    node_shares, shares = {}, []
    for nid, rows in pd.read_csv(paths.raw_photos_csv).groupby("node_id"):
        acc, share = node_term_shares(rows, cfg, paths, tax, by_bearing, ids)
        if acc is not None:
            node_shares[nid] = acc
            shares.append({"node_id": nid, **share})
    if not node_shares:
        print("no node has all its label maps yet; nothing to measure")
        return
    np.savez_compressed(paths.term_shares_by_bearing_npz, **node_shares)

    # ... and the class shares
    shares = pd.DataFrame(shares)
    shares.to_csv(paths.class_shares_csv, index=False)
    # a summary to judge the segmentation by: how much of the view each class takes, and how often it occurs
    classes = list(tax)
    summary = pd.DataFrame({"mean share": shares[classes].mean(), "nodes with > 0.5%": (shares[classes] > 0.005).mean()})
    print((summary * 100).round(1).to_string())
    print(f"measured {len(node_shares)} nodes; rows: {', '.join(by_bearing.rows)}")

    # 2. view indices
    nodes = gpd.read_file(paths.nodes_gpkg).to_crs(int(cfg["projected_crs"]))
    extra = sorted(set(node_shares) - set(nodes.node_id))
    if extra:
        sys.exit(f"{len(extra)} measured nodes are not among the sampling nodes (e.g. {extra[:3]}): "
                 f"the label maps are from other nodes")
    axis = street_axis(nodes)

    rows, empty = [], []
    for nid, p in node_shares.items():
        if not np.any(p[:by_bearing.row("weight")]):   # all term rows zero = no measurement, not an empty street
            empty.append(nid)
            continue
        gvi, vei = by_bearing.gvi_and_vei(p, 0.0, 360.0)
        s = by_bearing.shares(p, by_bearing.bin_mask(0.0, 360.0))
        rows.append({"node_id": nid, "street_axis_deg": axis.get(nid, np.nan), "GVI": gvi, "VEI": vei,
                     "GVI_eye_seg": 100 * s["vegetation_eye"], "SVF_band": s["sky"], "BVF_band": s["building"],
                     "PVI_band": s["V_pave"]})
    if empty:
        print(f"dropped {len(empty)} node(s) with no measured view: {', '.join(sorted(empty))}")

    keep = ["node_id", "osm_name", "typology", "street", "chain_pos_m", "geometry"]   # for the block faces
    m = gpd.GeoDataFrame(pd.DataFrame(rows).merge(nodes[keep], on="node_id"), geometry="geometry", crs=nodes.crs)

    # coverage audit: why each node without a metric has none
    missing = sorted(set(nodes.node_id) - set(node_shares))
    print(f"{len(nodes)} nodes, {len(node_shares)} measured, {len(missing)} without a metric")
    if missing:
        audit(paths, missing)

    # 3. block faces: a new face at every street change or gap > face_gap_m
    s = m.sort_values(["street", "chain_pos_m"])
    new = (s.street != s.street.shift()) | (s.chain_pos_m.diff().abs() > P["face_gap_m"])
    s["face_id"] = "f" + new.cumsum().astype(str)
    m["face_id"] = s.face_id.reindex(m.index)
    face = (m.groupby("face_id")
             .agg(osm_name=("osm_name", "first"), typology=("typology", "first"),
                  GVI=("GVI", "median"), VEI=("VEI", "median"), n_nodes=("node_id", "size"))
             .reset_index())
    face[face.n_nodes >= 2].to_csv(paths.block_faces_csv, index=False)
    print(f"block faces: {int((face.n_nodes >= 2).sum())}")

    # ... and Moran's I: how strongly neighbouring nodes resemble each other
    xy = np.c_[m.geometry.x.values, m.geometry.y.values]
    for col in ("GVI", "VEI"):
        ok = m[col].notna().values
        I, p = morans_i(m[col].values[ok], xy[ok], P["moran_radius_m"], P["moran_permutations"], cfg["seed"])
        print(f"Moran's I {col}: {I:+.3f} (p={p:.3f}); cluster standard errors on face_id")

    own = ["node_id", "street_axis_deg", "GVI", "GVI_eye_seg", "SVF_band", "BVF_band", "PVI_band", "VEI", "face_id"]
    m[own].to_csv(paths.view_indices_csv, index=False)   # node columns and geometry stay in nodes.gpkg
    print("wrote view-indices-per-node.csv, view-indices-per-block-face.csv")
