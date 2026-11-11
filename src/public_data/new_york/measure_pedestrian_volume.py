"""!
@package measure_pedestrian_volume
@brief The outcome per node: a modelled pedestrian volume, as the config describes it (`pedestrian_volume_per_node`;
NYCWalks).

The source is a vector file, or a zip holding one, in data/<city>/public_data/downloads/ (NYCWALKS,
downloaded the first time). `fields`
names the columns averaged into `value`; `observed_fields` maps each to one of observed counts on the same segments,
carried along as a check.

Steps, in this order:
  1. node_volume  per node, the length-weighted mean over the segments within radius_m (both sidewalks and, at a corner,
                  the crosswalks); value = the mean of the columns; the observed counts of any counted segment alongside
  2. run          reads the segments around the nodes (data_readers.pedestrian_volume_segments), writes
                  data/<city>/public_data/pedestrian-volume-per-node.csv, and the check against the observed counts in
                  pedestrian-volume-vs-counts.csv

Run by pipeline step measure_pedestrian_volume.
"""
import geopandas as gpd
import pandas as pd

from common.data_readers import pedestrian_volume_segments, record_map_layer
from common.downloads import download_file, record

# NYCWalks model estimates (Sevtsuk et al., Nature Cities 2025; MIT City Form Lab), pedestrians per hour per segment,
# 2018-2019. No licence stated: cite, do not redistribute (git-ignored). Downloaded the first time and logged.
NYCWALKS = {"file": "nycwalks-pedestrian-network-2018-2019.zip",
            "url": "https://www.dropbox.com/scl/fi/x3fgckcb120d4ugn561mv/SHP.zip?rlkey=tc4peg7avj3ibpese9tc0bx2d&dl=1"}


# ---------------------------------------------------------------- 1. node_volume
def node_volume(nodes, seg, columns, observed_fields, radius_m):
    """!
    @brief Each node's pedestrian volume: the length-weighted mean of the segments within radius_m.
    @param nodes            GeoDataFrame node_id, geometry (metre CRS).
    @param seg              The segments (same CRS) with the volume and observed-count columns.
    @param columns          The volume columns whose mean is `value`.
    @param observed_fields  {volume column: observed count column} (may be empty; entries for other columns are
                            ignored); a segment counts as counted where its count is above 0.
    @return DataFrame node_id, value, every volume column, n_segments, and observed_<volume column> (the
            length-weighted mean of the counted segments within radius_m; NaN where none was counted).
    """
    obs = {v: o for v, o in observed_fields.items() if v in columns and o in seg}
    buf = gpd.GeoDataFrame(nodes[["node_id"]], geometry=nodes.buffer(radius_m), crs=nodes.crs)
    part = gpd.overlay(seg[[*columns, *obs.values(), "geometry"]], buf, how="intersection",
                       keep_geom_type=True)
    part["w"] = part.length
    part = part[part.w > 0]

    def wmean(g, c):
        return (g[c] * g.w).sum() / g.w.sum()

    rows = []
    for nid, g in part.groupby("node_id"):
        r = {"node_id": nid, "n_segments": len(g), **{c: wmean(g, c) for c in columns}}
        for v, o in obs.items():
            k = g[g[o] > 0]
            r[f"observed_{v}"] = wmean(k, o) if len(k) else float("nan")
        rows.append(r)
    out = pd.DataFrame(rows)
    out.insert(1, "value", out[columns].mean(axis=1))
    return out


# ---------------------------------------------------------------- 2. run
def run(cfg, paths):
    """!
    @brief The modelled volume per node, and its check against the observed counts.
    """
    M = cfg["pedestrian_volume_per_node"]
    fields = M["fields"]
    src = paths.public_data_downloads / NYCWALKS["file"]
    if not src.exists():
        download_file(NYCWALKS["url"], src)
    record(paths, "nycwalks_pedestrian_volume", src, NYCWALKS["url"])
    record_map_layer(paths, "pedestrian_volume",
                     {"file": str(src.relative_to(paths.city_data)), "fields": fields, "name": M["name"]})
    nodes = gpd.read_file(paths.nodes_gpkg).to_crs(int(cfg["projected_crs"]))
    seg = pedestrian_volume_segments(src, nodes, M["radius_m"] + 50)        # 50 m beyond the radius: whole segments
    v = node_volume(nodes, seg, fields, M["observed_fields"], M["radius_m"])
    v.to_csv(paths.modelled_volume_csv, index=False)
    print(f"{M['name']}: {len(seg)} segments, volume for {len(v)} of {len(nodes)} nodes (median {v.value.median():.0f}, "
          f"mean of {', '.join(fields)})")
    cols = [f"observed_{c}" for c in fields if f"observed_{c}" in v]
    obs = v.dropna(subset=cols)
    o = obs[cols].mean(axis=1)
    check = {"periods": ", ".join(fields), "nodes": len(obs), "spearman": o.corr(obs.value, method="spearman"),
             "pearson": o.corr(obs.value)}
    pd.DataFrame([check]).to_csv(paths.volume_check_csv, index=False)
    print(f"  {len(obs)} nodes near an observed count in every period; modelled vs observed: Spearman "
          f"{check['spearman']:+.2f}, Pearson {check['pearson']:+.2f}")
