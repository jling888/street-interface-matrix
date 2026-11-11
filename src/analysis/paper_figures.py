"""Figures, written to outputs/<run_id>/analysis/figures/.

    sampling-nodes-unusable-crossed-out   the sampling frame on a map
    pca-weights-and-term-correlations     PCA weights, correlations between the terms
    tukey-street-type-differences         Tukey HSD differences with confidence intervals
    dwell-potential-map     the lingering predicted from M (map_dwell_potential)
    green-view-*            GVI and VEI context (segmentation measurements)
    maps, charts            below
    3d-*                    the footprints in 3D, with a column per node (map_style.buildings_3d_figures)
Every figure is drawn at its printed size (map_style.PAPER_RC: two-column 7.2 in or one-column 3.5 in wide, 300 dpi,
panels lettered); the interactive view is step make_results_viewer's. Each figure is skipped when its input does not
exist for this run.
Run by pipeline step make_figures.

Maps of every per-node dataset a run uses (all_maps):

    map-index-and-pillars        the three pillars and M per node
    map-rated-terms              every term the VLM rated, per node
    map-pedestrian-volume        the outcome: the modelled volume per street segment, and per node
    map-space-syntax             choice and integration at each radius
    map-land-use                 the lots by their main use (MapPLUTO), the subway entrances, land-use mix per node
    map-street-seating           the city's public seating and sidewalk cafes
    map-street-canyon            building height, street-wall height and H/W per node
    map-street-types             the street types (cluster_street_types) on the map
Each map is drawn in the config's projected_crs on the footprints and street centrelines, and skipped when its
input does not exist. Views are averaged per node.

Charts of a run's results (all_charts):

    ratings-distributions          each rated term's expected rung over the views (how far it spreads)
    ratings-validation             each check of a rating against independent data (Spearman rho)
    enclosure-rating-vs-hw         the enclosure rating against the measured H/W, with its band means
    dimension-coefficients         each dimension's elasticity with its 95% interval in every model
    model-fit                      R2 and AIC of the OLS models
    dimensions-vs-demand           each dimension and M against the outcome, per node
    indicator-factor-loadings      the indicators' loadings on three factors (do they group into their dimensions?)
    street-type-profiles           the street types' mean dimensions and M
Each chart is skipped when its input does not exist for this run.
"""
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from common.data_readers import (TERMS, footprints, map_layers, pedestrian_volume_segments, rated_fields, read_outcome,
                                 study_area)
from common.map_style import (DOUBLE_IN, DPI, GREY, INK, PAPER_RC, SINGLE_IN, SLOTS, buildings_3d_figures,
                              draw_nodes_map, panel, typology_colors)
from common.node_table import node_table, read_nodes

SIM_CMAP_STOPS = ["#F0DED6", "#E2D9B8", "#A9C4A0", "#4E8A72", "#1C5A49"]


def sim_cmap():
    return LinearSegmentedColormap.from_list("sim", SIM_CMAP_STOPS)


def save_current_figure(paths, name):
    plt.tight_layout()
    plt.savefig(paths.figures / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def sampling_nodes_map(cfg, paths):
    """The sampling frame on a map, every node unusable so far crossed out (step sample_street_nodes's map, for the paper)."""
    nodes = gpd.read_file(paths.nodes_gpkg)
    usable = nodes.node_id.map(read_nodes(paths).set_index("node_id").usable)
    draw_nodes_map(cfg, nodes, study_area(cfg["sample_osm_nodes"]["polygon"], paths.data_prep),
                   gpd.read_file(paths.streets_gpkg), set(nodes.node_id[~usable]),
                   paths.figures / "sampling-nodes-unusable-crossed-out.png")


def pca_and_correlations(cfg, paths):
    calc = pd.read_csv(paths.scores_csv)
    w = pd.read_csv(paths.pca_csv)
    terms = list(w.indicator)
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE_IN, max(3.0, 0.16 * len(terms) + 1.2)),
                           gridspec_kw={"width_ratios": [1, 1.25]})
    ax[0].barh((w.dimension + ": " + w.indicator.str[6:]).str.replace("_", " "), w.weight, color="#4E8A72")
    ax[0].invert_yaxis()
    ax[0].set_xlabel("PC1 weight (a diagnostic; equal weights used)")
    panel(ax[0], "A", "PC1 weights per dimension")
    r = calc[terms].corr()
    im = ax[1].imshow(r, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [t[6:].replace("_", " ") for t in terms]
    ax[1].set_xticks(range(len(terms)), labels, rotation=90)
    ax[1].set_yticks(range(len(terms)), labels)
    if len(terms) <= 12:                                    # numbers in the cells only while they stay legible
        for i in range(len(terms)):
            for j in range(len(terms)):
                ax[1].text(j, i, f"{r.iat[i, j]:.2f}", ha="center", va="center", fontsize=5)
    plt.colorbar(im, ax=ax[1], shrink=.8, label="Pearson r")
    panel(ax[1], "B", "Correlations between the indicators")
    save_current_figure(paths, "pca-weights-and-indicator-correlations")


def tukey_differences(paths):
    """One panel per tested variable, each on its own scale (volume, H/W and integration differ by orders)."""
    t = pd.read_csv(paths.tukey_csv)
    groups = list(t.groupby("variable", sort=False))
    fig, axes = plt.subplots(len(groups), 1, figsize=(SINGLE_IN * 1.3, 0.75 * len(groups) + 0.5), squeeze=False)
    for ax, (var, g) in zip(axes[:, 0], groups):
        yy = np.arange(len(g))[::-1]
        sig = g.reject.astype(str).eq("True").to_numpy()
        ax.errorbar(g.meandiff, yy, xerr=[g.meandiff - g.lower, g.upper - g.meandiff], fmt="none", color="#22406b")
        ax.scatter(g.meandiff, yy, s=12, color=np.where(sig, "#22406b", "white"), edgecolor="#22406b", zorder=3)
        ax.axvline(0, color="k", lw=.6)
        ax.set_ylim(-0.6, len(g) - 0.4)              # room for the first and last rows' markers
        ax.set_yticks(yy, [f"type {a} vs {b}" for a, b in zip(g.group1, g.group2)])
        panel(ax, "ABCDEFGH"[list(dict(groups)).index(var)], var.replace("_", " "))
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("mean difference between street types (Tukey HSD 95% interval; open dot: not significant)",
                  fontsize=6.5)
    fig.tight_layout()
    save_current_figure(paths, "tukey-street-type-differences")


def fig_dwell_potential(cfg, paths):
    d = pd.read_csv(paths.dwell_potential_grid_csv)
    cell = cfg["dwell_potential_map"]["cell_m"]
    ix, iy = np.rint((d.x - d.x.min()) / cell).astype(int), np.rint((d.y - d.y.min()) / cell).astype(int)
    Z = np.full((iy.max() + 1, ix.max() + 1), np.nan)
    Z[iy, ix] = d.P                                   # cells away from the streets stay blank
    fig, ax = plt.subplots(figsize=(SINGLE_IN + 0.5, SINGLE_IN))
    im = ax.imshow(Z, origin="lower", extent=(d.x.min(), d.x.max(), d.y.min(), d.y.max()), cmap=sim_cmap())
    plt.colorbar(im, ax=ax, shrink=.8, label="footfall x (M / 7), people per hour")
    ax.set_axis_off()
    ax.set_title("Predicted lingering (not measured dwell)", fontsize=7.5)
    save_current_figure(paths, "dwell-potential-map")


def context(cfg, paths, metrics):
    m = metrics.to_crs(4326)
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE_IN, 3.2))
    m.plot(column="GVI", cmap="YlGn", legend=True, markersize=2.5, ax=ax[0], legend_kwds={"shrink": 0.75})
    panel(ax[0], "A", "Green View Index (%), segmentation")
    v = m.VEI.dropna()
    if len(v) > 10:   # VEI bunches near 0.9: decile bands put the contrast where the data is
        m.plot(column="VEI", cmap="magma_r", norm=BoundaryNorm(np.unique(np.quantile(v, np.linspace(0, 1, 11))), 256),
               legend=True, markersize=2.5, ax=ax[1], legend_kwds={"shrink": 0.75})
    panel(ax[1], "B", "Visual Enclosure Index (decile bands)")
    for a in ax:
        a.set_aspect(1 / np.cos(np.radians(float(m.geometry.y.mean()))))
        a.set_axis_off()
    save_current_figure(paths, "green-view-and-enclosure-maps")
    if paths.block_faces_csv.exists():
        face = pd.read_csv(paths.block_faces_csv)
        fig, ax = plt.subplots(1, 2, figsize=(DOUBLE_IN, 2.9))
        for a, letter, df, unit in ((ax[0], "A", metrics, "nodes"), (ax[1], "B", face, "block faces")):
            for t in df.typology.value_counts().index:
                s = df[df.typology.eq(t)]
                a.scatter(s.VEI, s.GVI, s=3 if unit == "nodes" else 10, alpha=.6, c=typology_colors(cfg).get(t, GREY),
                          linewidths=0, label=f"{t.replace('_', ' ')} ({len(s)})")
            panel(a, letter, f"By street typology, {unit}")
            a.set_xlabel("Visual Enclosure Index")
            a.set_ylabel("Green View Index (%)")
            a.legend(frameon=False, markerscale=1.5)
            a.spines[["top", "right"]].set_visible(False)
        save_current_figure(paths, "green-view-vs-enclosure-scatter")



# ---------------------------------------------------------------- maps
PAD_M = 40                     # map extent: the nodes' bounding box plus this
LETTERS = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", *(f"A{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")]   # one per panel


def node_frame(cfg, paths):
    """!
    @brief One row per node: node_table() plus the run's view scores averaged per node, its outcome (volume),
           street type (the most common among its views) and lingering, where those exist.
    @return GeoDataFrame in the projected CRS.
    """
    nf = node_table(cfg, paths)
    if paths.scores_csv.exists():
        s = pd.read_csv(paths.scores_csv)
        cols = [c for c in s.select_dtypes("number") if c not in nf and c != "seq"]
        nf = nf.merge(s.groupby("node_id")[cols].mean().reset_index(), on="node_id", how="left")
    if paths.modelled_volume_csv.exists():
        nf = nf.merge(read_outcome(paths), on="node_id", how="left")
    if paths.clusters_csv.exists():
        c = pd.read_csv(paths.clusters_csv).groupby("node_id").cluster.agg(lambda x: x.mode().iloc[0])
        nf = nf.merge(c.rename("street_type").reset_index(), on="node_id", how="left")
    if paths.dwell_potential_nodes_csv.exists():
        d = pd.read_csv(paths.dwell_potential_nodes_csv)[["node_id", "dwell_potential"]]
        nf = nf.merge(d, on="node_id", how="left")
    return nf


def city_footprints(cfg, paths, nf):
    """!
    @brief The buildings with height_m in the projected CRS (the public data's buildings-with-height.gpkg), or None.
    """
    return footprints(paths)


def points(path, lat, lon, crs):
    """!
    @brief A points table (lat / lon columns) as a GeoDataFrame in `crs`.
    """
    d = pd.read_csv(path)
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d[lon], d[lat]), crs=4326).to_crs(crs)


class Basemap:
    """!
    @brief The study area's extent, footprints and street centrelines, drawn under every map.
    """

    def __init__(self, cfg, paths, nf):
        crs = int(cfg["projected_crs"])
        self.streets = gpd.read_file(paths.streets_gpkg).to_crs(crs)
        self.buildings = city_footprints(cfg, paths, nf)
        x0, y0, x1, y1 = nf.total_bounds
        self.extent = (x0 - PAD_M, x1 + PAD_M, y0 - PAD_M, y1 + PAD_M)
        self.aspect = (self.extent[3] - self.extent[2]) / (self.extent[1] - self.extent[0])

    def figure(self, rows, cols, width=DOUBLE_IN, bar=True):
        """!
        @brief A grid of map panels sized to the study area's shape, each with room for a colour bar.
        @return (fig, axes as a flat list).
        """
        w = width / cols * (0.8 if bar else 1.0)
        fig, axes = plt.subplots(rows, cols, figsize=(width, rows * (w * self.aspect + 0.3)), layout="constrained",
                                 squeeze=False)
        return fig, list(axes.flat)

    def draw(self, ax, letter, title, buildings=True):
        if buildings and self.buildings is not None:
            self.buildings.plot(ax=ax, color="#e6e4df", edgecolor="#d2cfc8", lw=0.1, zorder=0)
        self.streets.plot(ax=ax, color="#bdbab3", lw=0.4, zorder=1)
        ax.set_xlim(*self.extent[:2])
        ax.set_ylim(*self.extent[2:])
        ax.set_aspect("equal")
        ax.set_axis_off()
        panel(ax, letter, title)


def dots(ax, nf, col, cmap="viridis", norm=None, label=None, s=3):
    """!
    @brief The nodes coloured by `col` (NaN nodes left out), with a labelled colour bar.
    """
    d = nf[nf[col].notna()]
    sc = ax.scatter(d.geometry.x, d.geometry.y, c=d[col], s=s, cmap=cmap, norm=norm, zorder=3, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, aspect=25, pad=0.01)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    if label:
        cb.set_label(label, fontsize=6.5)


def save(fig, paths, name):
    fig.savefig(paths.figures / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)


def map_index_and_dimensions(cfg, paths, nf, base, cmap):
    names = cfg["metric"]["names"]
    fig, axes = base.figure(2, 2)
    for ax, letter, (col, title) in zip(axes, LETTERS, [*names.items(), ("M", "M, the geometric mean of the pillars")]):
        base.draw(ax, letter, title)
        dots(ax, nf, col, cmap, label="score, 1-7")
    save(fig, paths, "map-index-and-dimensions")


def map_rated_terms(cfg, paths, nf, base, cmap):
    fields = rated_fields(pd.read_csv(paths.ratings_csv, nrows=0))
    # a term once, on its 0-1 scale: the rating's own column holds the rounded rung
    terms = [TERMS.get(f, f) for f in fields if TERMS.get(f, f) in nf]
    if not terms:
        return
    cols = 4
    fig, axes = base.figure(-(-len(terms) // cols), cols)
    for ax, letter, t in zip(axes, LETTERS, terms):
        base.draw(ax, letter, t.replace("_", " "), buildings=False)
        dots(ax, nf, t, cmap, s=1.5)
    for ax in axes[len(terms):]:
        ax.set_axis_off()
    save(fig, paths, "map-rated-terms")


def map_pedestrian_volume(cfg, paths, nf, base):
    M = map_layers(paths).get("pedestrian_volume")
    path = paths.city_data / M["file"] if M else None
    fig, axes = base.figure(1, 2)
    if path and path.exists():
        seg = pedestrian_volume_segments(path, nf, PAD_M)
        seg["volume"] = seg[M["fields"]].mean(axis=1)
        seg = seg[seg.volume > 0]
        base.draw(axes[0], "A", f"{M['name']} volume per segment")
        lc = seg.plot(ax=axes[0], column="volume", cmap="magma_r", lw=1.0, zorder=2,
                      norm=LogNorm(seg.volume.min(), seg.volume.max()))
        cb = plt.colorbar(lc.collections[-1], ax=axes[0], shrink=0.75, aspect=25, pad=0.01)
        cb.set_label("people per hour", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    else:
        axes[0].set_axis_off()
    base.draw(axes[1], "B", "Outcome per node")
    if "volume" in nf:
        v = nf.volume.where(nf.volume > 0)
        dots(axes[1], nf.assign(volume=v), "volume", "magma_r", LogNorm(v.min(), v.max()), "people per hour")
    save(fig, paths, "map-pedestrian-volume")


def map_space_syntax(cfg, paths, nf, base):
    N = cfg["space_syntax_network"]
    radii = sorted([N["radius_m"], *N.get("extra_radii_m", [])])
    col = lambda kind, r: kind if r == N["radius_m"] else f"{kind}_{r}"
    if not all(col(k, r) in nf for k in ("choice", "integration") for r in radii):
        return
    fig, axes = base.figure(2, len(radii))
    for j, r in enumerate(radii):
        base.draw(axes[j], LETTERS[j], f"Choice, {r:,} m")
        dots(axes[j], nf.assign(v=np.log1p(nf[col("choice", r)])), "v", "cividis", label="ln(1 + choice)")
        k = len(radii) + j
        base.draw(axes[k], LETTERS[k], f"Integration, {r:,} m")
        dots(axes[k], nf.assign(v=nf[col("integration", r)] * 1e3), "v", "cividis", label="closeness x 1,000")
    save(fig, paths, "map-space-syntax")


def map_land_use(cfg, paths, nf, base):
    layers = map_layers(paths)
    P = layers.get("lots")
    if not P or not (paths.city_data / P["file"]).exists():
        return
    lots = points(paths.city_data / P["file"], P["lat_field"], P["lon_field"], nf.crs)
    x0, x1, y0, y1 = base.extent
    lots = lots.cx[x0:x1, y0:y1]
    F = P["fields"]
    lots[list(F.values())] = lots[list(F.values())].fillna(0)      # a lot without a use's area has none of it
    uses = {k: F[k] for k in ("retail", "office", "residential")}
    area = lots[list(uses.values())].clip(lower=0)
    rest = (lots[F["total"]] - area.sum(axis=1)).clip(lower=0)
    main = pd.concat([area, rest.rename("other")], axis=1).idxmax(axis=1)
    colors = dict(zip(uses.values(), SLOTS)) | {"other": GREY}
    fig, axes = base.figure(1, 2)
    base.draw(axes[0], "A", "Lots by largest use; subway entrances")
    for key, name in [*((v, k) for k, v in uses.items()), ("other", "other")]:
        s = lots[main.eq(key) & lots[F["total"]].gt(0)]
        axes[0].scatter(s.geometry.x, s.geometry.y, s=np.sqrt(s[F["total"]]) / 40, c=colors[key], alpha=0.75,
                        linewidths=0, label=f"{name} ({len(s):,})", zorder=2)
    T = layers.get("transit")
    if T and (paths.city_data / T["file"]).exists():
        sub = points(paths.city_data / T["file"], T["lat_field"], T["lon_field"], nf.crs).cx[x0:x1, y0:y1]
        axes[0].scatter(sub.geometry.x, sub.geometry.y, marker="s", s=5, c="black", label="subway entrance", zorder=3)
    axes[0].legend(fontsize=5.5, loc="lower right", markerscale=1.5, handletextpad=0.2, borderpad=0.3)
    base.draw(axes[1], "B", "Land-use mix per node")
    if "land_use_mix" in nf:
        dots(axes[1], nf, "land_use_mix", "viridis", label="mix (0 = one use, 1 = even)")
    save(fig, paths, "map-land-use")


def map_street_seating(cfg, paths, nf, base):
    S = map_layers(paths).get("seating") or {}
    have = {n: s for n, s in S.items() if (paths.city_data / s["file"]).exists()}
    if not have:
        return
    x0, x1, y0, y1 = base.extent
    fig, axes = base.figure(1, 1, SINGLE_IN, bar=False)
    base.draw(axes[0], "", "The city's records of places to sit")
    for (name, s), color in zip(have.items(), SLOTS):
        p = points(paths.city_data / s["file"], s["lat_field"], s["lon_field"], nf.crs).cx[x0:x1, y0:y1]
        axes[0].scatter(p.geometry.x, p.geometry.y, s=4, c=color, label=f"{name.replace('_', ' ')} ({len(p):,})",
                        zorder=3, linewidths=0)
    axes[0].legend(fontsize=6, loc="lower right", markerscale=2)
    save(fig, paths, "map-street-seating")


def map_street_canyon(cfg, paths, nf, base):
    if "HW_effective" not in nf:
        return
    fig, axes = base.figure(1, 3)
    base.draw(axes[0], "A", "Building height", buildings=False)
    if base.buildings is not None:
        b = base.buildings
        b.plot(ax=axes[0], column="height_m", cmap="YlOrBr", norm=LogNorm(5, b.height_m.max()), lw=0, zorder=2)
        sm = plt.cm.ScalarMappable(LogNorm(5, b.height_m.max()), "YlOrBr")
        cb = plt.colorbar(sm, ax=axes[0], shrink=0.75, aspect=25, pad=0.01)
        cb.set_label("roof height (m)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    base.draw(axes[1], "B", "Street-wall height H")
    dots(axes[1], nf, "H_m", "YlOrBr", label="m")
    base.draw(axes[2], "C", "Height to width H/W")
    dots(axes[2], nf.assign(hw=nf.HW_effective.where(nf.HW_effective > 0)), "hw", "YlOrBr", LogNorm(0.2, 10), "H/W")
    save(fig, paths, "map-street-canyon")


def map_street_types(cfg, paths, nf, base):
    if "street_type" not in nf:
        return
    cen = pd.read_csv(paths.centroids_csv).set_index("cluster") if paths.centroids_csv.exists() else None
    names = cfg["metric"]["names"]
    fig, axes = base.figure(1, 1, SINGLE_IN, bar=False)
    base.draw(axes[0], "", "Street types (k-means on the three dimensions)")
    for k, color in zip(sorted(nf.street_type.dropna().unique()), SLOTS + [GREY] * 9):
        s = nf[nf.street_type.eq(k)]
        lab = f"type {int(k)}, {len(s)} nodes"
        if cen is not None and k in cen.index:
            c = cen.loc[k]
            lab += " (" + ", ".join(f"{n} {c[k]:.1f}" for k, n in names.items() if k in c) + ")"
        axes[0].scatter(s.geometry.x, s.geometry.y, s=3, c=color, label=lab, zorder=3, linewidths=0)
    axes[0].legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), markerscale=2.5, frameon=False)
    save(fig, paths, "map-street-types")


def all_maps(cfg, paths, cmap):
    """!
    @brief Every map whose input exists for this run.
    @param cmap  The figures' colour map for the index's scores.
    @return (node_frame(), the Basemap), for the charts and the 3D figures.
    """
    nf = node_frame(cfg, paths)
    base = Basemap(cfg, paths, nf)
    if "M" in nf:
        map_index_and_dimensions(cfg, paths, nf, base, cmap)
        map_rated_terms(cfg, paths, nf, base, cmap)
    map_pedestrian_volume(cfg, paths, nf, base)
    map_space_syntax(cfg, paths, nf, base)
    map_land_use(cfg, paths, nf, base)
    map_street_seating(cfg, paths, nf, base)
    map_street_canyon(cfg, paths, nf, base)
    map_street_types(cfg, paths, nf, base)
    return nf, base


# ---------------------------------------------------------------- charts
ACCENT = SLOTS[0]




def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def ratings_distributions(cfg, paths):
    d = pd.read_csv(paths.ratings_csv)
    terms = rated_fields(d)
    cols = 4
    rows = -(-len(terms) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE_IN, 1.25 * rows + 0.3), sharex=True, layout="constrained")
    for ax, letter, t in zip(axes.flat, LETTERS, terms):
        v = d[t + "_ev"].dropna()
        ax.hist(v, bins=np.arange(1, 7.26, 0.25), color=GREY)
        ax.axvline(v.mean(), color=ACCENT, lw=1)
        panel(ax, letter, t.replace("_", " "))
        ax.text(0.98, 0.95, f"mean {v.mean():.1f}\nsd {v.std():.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=5.5)
        ax.set_yticks([])
        ax.set_xticks(range(1, 8))
        tidy(ax)
    for ax in list(axes.flat)[len(terms):]:
        ax.set_axis_off()
    fig.supxlabel("expected rung (1-7); line = mean", fontsize=7)
    save(fig, paths, "ratings-distributions")


def ratings_validation(paths):
    rows = []
    if paths.anchor_csv.exists():
        rows += [(f"{r.field} vs segmented {r.counterpart}", r.rho, "segmentation")
                 for r in pd.read_csv(paths.anchor_csv).itertuples()]
    if paths.street_seating_check_csv.exists():
        rows += [(f"{r.field} vs {r.record}", r.rho, "city records")
                 for r in pd.read_csv(paths.street_seating_check_csv).itertuples()]
    if paths.city_records_check_csv.exists():
        rows += [(f"{r.field} vs {r.record}", r.rho, "segmentation" if r.record.startswith("segmented") else "city records")
                 for r in pd.read_csv(paths.city_records_check_csv).itertuples()]
    if paths.enclosure_vs_geometry_csv.exists():
        rows += [(f"enclosure vs {r.measure}", r.rho, "footprints")
                 for r in pd.read_csv(paths.enclosure_vs_geometry_csv).itertuples()]
    if paths.physical_records_check_csv.exists():
        rows += [(f"{r.field} vs {r.record}", r.rho, "physical records")
                 for r in pd.read_csv(paths.physical_records_check_csv).itertuples()]
    if not rows:
        return
    t = pd.DataFrame(rows, columns=["check", "rho", "source"])
    t["check"] = t.check.str.replace("_", " ")
    colors = dict(zip(["segmentation", "city records", "footprints", "physical records"], [*SLOTS, GREY]))
    fig, ax = plt.subplots(figsize=(DOUBLE_IN * 0.75, 0.16 * len(t) + 0.6), layout="constrained")
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.rho, color=t.source.map(colors), height=0.7)
    for yy, r in zip(y, t.rho):
        right = r >= -0.05                          # near-zero labels go right, clear of the term names
        ax.text(max(r, 0) + 0.01 if right else r - 0.01, yy, f"{r:.2f}", va="center", ha="left" if right else "right",
                fontsize=5.5)
    ax.set_yticks(y, t.check, fontsize=6)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlim(min(t.rho.min() - 0.12, -0.05), None)   # room for a negative bar's label
    ax.legend(handles=[Patch(color=c, label=s) for s, c in colors.items()], loc="lower right", frameon=False)
    ax.set_xlabel("Spearman rho, rating vs independent measure")
    tidy(ax)
    save(fig, paths, "ratings-validation")


def enclosure_vs_hw(paths, nf):
    if "HW_effective" not in nf or not paths.ratings_csv.exists():
        return
    d = pd.read_csv(paths.ratings_csv)[["node_id", "enclosure_ev"]].merge(
        pd.DataFrame(nf[["node_id", "HW_effective"]]), on="node_id").dropna()
    d = d[d.HW_effective > 0]
    fig, ax = plt.subplots(figsize=(SINGLE_IN, 2.5), layout="constrained")
    ax.scatter(d.HW_effective, d.enclosure_ev, s=1.5, color=GREY, alpha=0.5, linewidths=0, label="views")
    if paths.enclosure_curve_csv.exists():
        c = pd.read_csv(paths.enclosure_curve_csv).query("what == 'band'")
        mid = c.label.str.strip("(]").str.split(", ").apply(lambda p: np.sqrt(max(float(p[0]), 0.25) * float(p[1])))
        ax.plot(mid, c.mean_rating, "o-", color=ACCENT, ms=3, label="mean per H/W band")
    ax.axvline(0.5, color=INK, ls="--", lw=0.6)
    ax.text(0.53, 1.3, "H/W 0.5: rooftops leave\na level photo", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel("measured H/W (footprints)")
    ax.set_ylabel("enclosure rating (1-7)")
    ax.set_ylim(1, 7)
    ax.legend(loc="upper left", frameon=False)
    tidy(ax)
    save(fig, paths, "enclosure-rating-vs-hw")


def enclosure_by_height(paths):
    """!
    @brief Two panels, one measure each, over the same bands of building height.
    """
    if not paths.enclosure_by_height_csv.exists():
        return
    h = pd.read_csv(paths.enclosure_by_height_csv)
    x = np.arange(len(h))
    ticks = h.storeys.str.strip("(]").str.replace(", ", "-").str.replace(".0", "").str.replace("-100", "+")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_IN, 2.2), layout="constrained", sharex=True)
    for ax, letter, col, title, ylabel, scale in ((axes[0], "A", "enclosure_rating", "Enclosure rating", "mean rating (1-7)", 1),
                                                  (axes[1], "B", "visible_sky_share", "Sky in the level photos", "share of the view (%)", 100)):
        ax.plot(x, h[col] * scale, "o-", color=ACCENT, ms=3, lw=1.2)
        panel(ax, letter, title)
        ax.set_xticks(x, ticks)
        ax.set_xlabel("building height (storeys)")
        ax.set_ylabel(ylabel)
        tidy(ax)
    axes[0].set_ylim(1, 7)
    axes[1].set_ylim(0, None)
    save(fig, paths, "enclosure-by-height")


def dimension_coefficients(cfg, paths):
    c = pd.read_csv(paths.coefficients_csv)
    terms = {f"ln_{k}": name for k, name in cfg["metric"]["names"].items()}
    c = c[c.term.isin(terms) & (c.model.str.startswith("OLS Model") | c.model.str.contains("nodes"))
          & ~c.model.str.contains("indicators")].copy()
    se = c.se_clustered.where(c.se_clustered.notna(), c.b / c.t)
    c["lo"], c["hi"] = c.b - 1.96 * se.abs(), c.b + 1.96 * se.abs()
    models = list(dict.fromkeys(c.model))
    fig, axes = plt.subplots(1, len(terms), figsize=(DOUBLE_IN, 0.2 * len(models) + 0.8), sharey=True, layout="constrained")
    y = np.arange(len(models))[::-1]
    for ax, letter, (term, name) in zip(axes, LETTERS, terms.items()):
        g = c[c.term == term].set_index("model").reindex(models)
        sig = (g.lo > 0) | (g.hi < 0)
        ax.errorbar(g.b, y, xerr=[g.b - g.lo, g.hi - g.b], fmt="none", color=GREY, lw=0.8)
        ax.scatter(g.b, y, s=12, color=np.where(sig, ACCENT, "white"), edgecolor=ACCENT, lw=0.7, zorder=3)
        ax.axvline(0, color=INK, lw=0.6)
        panel(ax, letter, name)
        ax.set_xlabel("elasticity b (95% CI)")
        tidy(ax)
    axes[0].set_yticks(y, [m.replace(" (nodes)", ", per node") for m in models])
    save(fig, paths, "dimension-coefficients")


def model_fit(paths):
    f = pd.read_csv(paths.model_fit_csv)
    f = f[f.model.str.startswith("OLS")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_IN, 0.2 * len(f) + 0.7), sharey=True, layout="constrained")
    y = np.arange(len(f))[::-1]
    best = f.aic.idxmin()
    for ax, letter, col, title in ((axes[0], "A", "r2", "R² (share of demand explained)"),
                                   (axes[1], "B", "aic", "AIC (lower is better)")):
        ax.barh(y, f[col], color=[ACCENT if i == best else GREY for i in f.index], height=0.65)
        for yy, v in zip(y, f[col]):
            ax.text(v, yy, f" {v:.2f}" if col == "r2" else f" {v:,.0f}", va="center", fontsize=5.5)
        panel(ax, letter, title)
        ax.set_xlim(0, f[col].max() * 1.18)
        tidy(ax)
    axes[0].set_yticks(y, f.model.str.replace("OLS ", ""))
    save(fig, paths, "model-fit")


def dimensions_vs_demand(cfg, paths, nf):
    if "volume" not in nf or "M" not in nf:
        return
    cols = [*cfg["metric"]["names"].items(), ("M", "M")]
    d = nf[nf.volume.gt(0) & nf.M.notna()]
    fig, axes = plt.subplots(1, len(cols), figsize=(DOUBLE_IN, 1.9), sharey=True, layout="constrained")
    for ax, letter, (col, name) in zip(axes, LETTERS, cols):
        ax.scatter(d[col], d.volume, s=1.5, color=ACCENT if col == "M" else GREY, alpha=0.6, linewidths=0)
        panel(ax, letter, f"{name}, ρ {spearmanr(d[col], d.volume).statistic:+.2f}")
        ax.set_xlabel(f"{name} (1-7)")
        tidy(ax)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("people per hour")
    save(fig, paths, "dimensions-vs-demand")


def term_factor_loadings(cfg, paths):
    f = pd.read_csv(paths.indicator_factors_csv)
    names = cfg["metric"]["names"]
    F = [c for c in f if c.startswith("F")]
    fig, ax = plt.subplots(figsize=(SINGLE_IN, 0.2 * len(f) + 0.6), layout="constrained")
    im = ax.imshow(f[F], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(f)):
        for j, c in enumerate(F):
            ax.text(j, i, f"{f[c].iat[i]:.2f}", ha="center", va="center", fontsize=5.5)
    ax.set_xticks(range(len(F)), [f"factor {c[1:]}" for c in F])
    ax.set_yticks(range(len(f)), [f"{t[6:].replace('_', ' ')} ({names[p]})" for t, p in zip(f.indicator, f.dimension)])
    cb = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25)
    cb.set_label("loading", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    save(fig, paths, "indicator-factor-loadings")


def street_type_profiles(cfg, paths):
    c = pd.read_csv(paths.centroids_csv)
    cols = [*cfg["metric"]["names"].items(), ("M", "M")]
    fig, ax = plt.subplots(figsize=(SINGLE_IN, 2.3), layout="constrained")
    w = 0.8 / len(cols)
    for i, ((col, name), a) in enumerate(zip(cols, [0.35, 0.55, 0.75, 1.0])):
        ax.bar(np.arange(len(c)) + (i - (len(cols) - 1) / 2) * w, c[col], w, color=ACCENT, alpha=a, label=name)
    ax.set_xticks(np.arange(len(c)), [f"type {int(k)}\n{int(n)} views\n{np.exp(v):,.0f} people/h"
                                      for k, n, v in zip(c.cluster, c.views, c.ln_volume)], fontsize=6)
    ax.set_ylim(1, 7)
    ax.set_ylabel("mean score (1-7)")
    ax.legend(ncol=4, loc="upper center", frameon=False, handlelength=1, columnspacing=0.8)
    tidy(ax)
    save(fig, paths, "street-type-profiles")


def all_charts(cfg, paths, nf):
    """!
    @brief Every chart whose input exists for this run.
    @param nf  node_frame().
    """
    if paths.ratings_csv.exists():
        ratings_distributions(cfg, paths)
    ratings_validation(paths)
    enclosure_vs_hw(paths, nf)
    enclosure_by_height(paths)
    if paths.coefficients_csv.exists():
        dimension_coefficients(cfg, paths)
    if paths.model_fit_csv.exists():
        model_fit(paths)
    dimensions_vs_demand(cfg, paths, nf)
    if paths.indicator_factors_csv.exists():
        term_factor_loadings(cfg, paths)
    if paths.centroids_csv.exists():
        street_type_profiles(cfg, paths)


# ---------------------------------------------------------------- run
def run(cfg, paths):
    """!
    @brief Every figure whose input exists for this run (the folder's earlier figures removed first).
    """
    with plt.rc_context(PAPER_RC):              # every figure at its printed size
        paths.figures.mkdir(parents=True, exist_ok=True)
        for p in [*paths.figures.glob("*.png"), *paths.figures.glob("*.html")]:
            p.unlink()
        sampling_nodes_map(cfg, paths)
        if paths.scores_csv.exists():
            pca_and_correlations(cfg, paths)
        if paths.tukey_csv.exists() and len(pd.read_csv(paths.tukey_csv)):
            tukey_differences(paths)
        if paths.dwell_potential_grid_csv.exists():
            fig_dwell_potential(cfg, paths)
        if paths.view_indices_csv.exists():
            table = node_table(cfg, paths)
            context(cfg, paths, table[table.GVI.notna()])
        nf, base = all_maps(cfg, paths, sim_cmap())
        all_charts(cfg, paths, nf)
        buildings_3d_figures(paths, nf, base, sim_cmap())
    for p in sorted([*paths.figures.glob("*.png"), *paths.figures.glob("*.html")]):
        print(f"  {p.name}")
