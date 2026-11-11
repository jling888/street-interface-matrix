"""!
@package node_table
@brief The node table: the sampling nodes with their usable flags, joined with every per-node measurement.

One writer per file; what several steps need is joined here on node_id when it is read.
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd


def node_review(paths):
    """!
    @brief Hand-made decisions on individual nodes: data/<city>/data_prep/node-review-decisions.csv (optional).
    @return DataFrame node_id, decision, reason. decision: "ok" (looked at, no change), "exclude", or
            "keep" (overrides every automatic check). Written by the review app.
    """
    path = paths.node_review
    if not path.exists():
        return pd.DataFrame({"node_id": [], "decision": [], "reason": []}, dtype=str)
    review = pd.read_csv(path, dtype=str).fillna("")
    bad = sorted(set(review.decision) - {"ok", "exclude", "keep"})
    if bad:
        sys.exit(f"{path}: decision must be ok, exclude or keep, not {bad}")
    return review


def read_nodes(paths):
    """!
    @brief The sampling nodes (nodes.csv) with usable / exclude_reason, by one explicit rule.

    A node is usable when every check that has run passes; the first failing check gives the reason:
      1. not excluded in node-review-decisions.csv (hand-made, in the review app)
      2. a usable panorama was found     (street-view-metadata.csv: reason, "" if usable)
    A node-review-decisions.csv `keep` overrides every check.

    @return DataFrame: nodes.csv columns plus usable (bool) and exclude_reason ("" if usable).
    """
    n = pd.read_csv(paths.nodes_csv)
    review = node_review(paths)
    absent = sorted(set(review.node_id) - set(n.node_id))
    if absent:
        sys.exit(f"{paths.node_review} lists ids not in this frame: {absent[:8]}")
    excluded = review[review.decision == "exclude"]
    checks = [n.node_id.map(dict(zip(excluded.node_id, excluded.reason.replace("", "excluded by review"))))]

    if paths.pano_metadata_csv.exists():
        m = pd.read_csv(paths.pano_metadata_csv).set_index("node_id")
        if "reason" in m:                                  # "" (read as NaN) = usable imagery
            checks.append(n.node_id.map(m.reason.dropna()).where(n.node_id.isin(m.index),
                                                                  "no imagery metadata (rerun download_street_view)"))

    reason = pd.concat(checks, axis=1).bfill(axis=1).iloc[:, 0]     # first failing check
    reason[n.node_id.isin(review.node_id[review.decision == "keep"])] = np.nan
    return n.assign(usable=reason.isna(), exclude_reason=reason.fillna(""))


def node_table(cfg, paths):
    """!
    @brief Everything known per node: the nodes with their geometry, the usable flags, and the columns
           of every per-node measurement file that exists (view indices, street height / width, Space Syntax,
           land use, city records).
    @return GeoDataFrame in the config's projected_crs, one row per node; measurements not yet made are
            absent columns, nodes a step did not measure are NaN.
    """
    table = gpd.read_file(paths.nodes_gpkg).to_crs(int(cfg["projected_crs"])).merge(
        read_nodes(paths)[["node_id", "usable", "exclude_reason"]], on="node_id")
    for path in (paths.view_indices_csv, paths.street_canyons_csv, paths.space_syntax_csv, paths.land_use_csv,
                 paths.city_records_csv):
        if path.exists():
            m = pd.read_csv(path)
            table = table.merge(m[["node_id", *[c for c in m if c not in table]]], on="node_id", how="left")
    return table


def street_axis(nodes, k=5):
    """!
    @brief Street bearing mod 180 per node: the principal direction of its k nearest nodes on the same street.
    @return {node_id: bearing in degrees}; NaN for a node alone on its street.
    """
    out = {}
    for _, grp in nodes.groupby("street"):
        P = np.c_[grp.geometry.x.values, grp.geometry.y.values]
        for i, nid in enumerate(grp.node_id.values):
            near = P[np.argsort(np.linalg.norm(P - P[i], axis=1))[:min(k, len(P))]]
            if len(near) < 2:
                out[nid] = np.nan
                continue
            v = np.linalg.svd(near - near.mean(0), full_matrices=False)[2][0]
            out[nid] = float(np.degrees(np.arctan2(v[0], v[1])) % 180)
    return out
