"""!
@package dwell_potential_map
@brief Where the Street Interface Matrix predicts people to linger, as a map: a prediction, not a measurement.

    P(x, y) = sum_i L_i * K_i(x, y) / sum_i K_i(x, y)        K_i = exp(-d_i(x, y)^2 / (2 R^2))
    L_i = V_i * (M_i / 7)^s      footfall (the node's modelled pedestrians per hour, step measure_pedestrian_volume's
    outcome) x the stopping
                                 score M_i / 7 on 0-1

No lingering data exists to estimate s, so L is given at each s of dwell_potential.lingering_exponents
(lingering_s<s> per node); dwell_potential, the one mapped, is s = 1. A kernel-weighted mean, so the map does not grow
where nodes cluster (street corners); cells with no node within mask_m are left out. Writes
outputs/<run_id>/analysis/dwell-potential/: dwell-potential-per-node.csv,
dwell-potential-per-street.csv (the node means by street and by street type) and dwell-potential-grid.csv (x, y, P on a
cell_m grid, in the projected CRS). Run by pipeline step map_dwell_potential.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common.data_readers import read_outcome
from common.node_table import read_nodes


def potential_surface(x, y, value, R, cell, mask_m):
    """!
    @brief The Gaussian-kernel-weighted mean of `value` at points (x, y), on a regular grid.
    @return (X, Y, P); P is NaN where no point lies within mask_m.
    """
    gx = np.arange(x.min() - mask_m, x.max() + mask_m + cell, cell)
    gy = np.arange(y.min() - mask_m, y.max() + mask_m + cell, cell)
    X, Y = np.meshgrid(gx, gy)
    num, den = np.zeros_like(X), np.zeros_like(X)
    for xi, yi, vi in zip(x, y, value):
        k = np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2 * R ** 2))
        num += vi * k
        den += k
    near = cKDTree(np.c_[x, y]).query(np.c_[X.ravel(), Y.ravel()], distance_upper_bound=mask_m)[0]
    P = np.where(np.isfinite(near).reshape(X.shape), num / np.maximum(den, 1e-300), np.nan)
    return X, Y, P


def run(cfg, paths):
    """!
    @brief The lingering map from the nodes' footfall and M.
    """
    P = cfg["dwell_potential_map"]
    node = pd.read_csv(paths.scores_csv).groupby("node_id").M.mean().rename("score").to_frame()
    vol = read_outcome(paths)
    node = node.join(vol.set_index("node_id").volume.rename("footfall"), how="inner")
    for s in P["lingering_exponents"]:
        node[f"lingering_s{s:g}"] = node.footfall * (node.score / 7) ** s
    node["dwell_potential"] = node.footfall * node.score / 7
    nodes = read_nodes(paths).set_index("node_id")
    g = gpd.read_file(paths.nodes_gpkg).to_crs(int(cfg["projected_crs"])).set_index("node_id").loc[node.index].geometry
    out = node.assign(x=g.x.to_numpy(), y=g.y.to_numpy())
    out.reset_index().to_csv(paths.dwell_potential_nodes_csv, index=False)
    by = out.join(nodes[["osm_name", "typology"]])
    tab = pd.concat([by.groupby(k).dwell_potential.agg(["count", "mean", "std", "min", "max"])
                     .rename_axis("name").reset_index().assign(by=k) for k in ("osm_name", "typology")])
    tab = tab.rename(columns={"count": "nodes"}).sort_values(["by", "mean"], ascending=[True, False])
    tab[["by", "name", "nodes", "mean", "std", "min", "max"]].to_csv(paths.dwell_by_street_csv, index=False)

    X, Y, Pxy = potential_surface(out.x.to_numpy(), out.y.to_numpy(), out.dwell_potential.to_numpy(),
                                  P["bandwidth_m"], P["cell_m"], P["mask_m"])
    keep = np.isfinite(Pxy.ravel())
    pd.DataFrame({"x": X.ravel()[keep], "y": Y.ravel()[keep], "P": Pxy.ravel()[keep]}).to_csv(
        paths.dwell_potential_grid_csv, index=False)
    print(f"lingering for {len(out)} nodes, median {out.dwell_potential.median():.2f}; grid {X.shape}, "
          f"{keep.sum():,} cells within {P['mask_m']} m of a node")
