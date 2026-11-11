"""!
@package map_style
@brief Maps of the study area on an OpenStreetMap background, the colours of street types, and the figure style.

The study area in 3D from the building footprints and their heights, written by step make_figures
(buildings_3d_figures):

    3d-buildings-by-height.png             the footprints extruded to their roof height, coloured by height
    3d-footfall-columns-coloured-by-m.png  the buildings in grey, a column at every node: height = its pedestrian
                                           volume, colour = M
Drawn two-column wide at 300 dpi (PAPER_RC). The interactive 3D view is the results viewer's (step
make_results_viewer). Skipped without footprints. Buildings and columns are one collection in the PNGs, so
matplotlib sorts their faces together (two collections would be drawn one over the other whole).
"""
import re

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Categorical slots validated for maps (all pairs: CVD dE 9.2, normal-vision dE 24.0); past three, grey.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
GREY, INK, MUTED = "#a3a29c", "#0b0b0b", "#52514e"

# The paper's figure style: drawn at the printed size (two-column 180 mm, one-column 89 mm), so 6-8 pt text stays
# 6-8 pt in print; saved at 300 dpi
DOUBLE_IN, SINGLE_IN = 7.2, 3.5
DPI = 300
PAPER_RC = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "figure.titlesize": 8, "axes.linewidth": 0.6,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0, "savefig.dpi": DPI}


def panel(ax, letter, title):
    """!
    @brief A panel's title, led by its letter (A, B, ...) for the caption to refer to.
    """
    ax.set_title(f"{letter}   {title}", loc="left", fontsize=7.5)


def typology_colors(cfg):
    """!
    @brief A colour per street type, in the city's `typology` order: three slots, then grey.
    @return {typology label: hex colour}, always including 'other' (the first slot if the city has no types).
    """
    names = list(cfg["sample_osm_nodes"].get("typology") or {})
    return {**{n: SLOTS[i] if i < len(SLOTS) else GREY for i, n in enumerate(names)},
            "other": GREY if names else SLOTS[0]}


NAVY = "#1f2a37"
ABBREV = [(r"\bEast\b", "E"), (r"\bWest\b", "W"), (r"\bNorth\b", "N"), (r"\bSouth\b", "S"),
          (r"\bStreet\b", "St"), (r"\bAvenue\b", "Ave"), (r"\bPlace\b", "Pl"), (r"\bRoad\b", "Rd")]


def _label(name):
    """!
    @brief A street name abbreviated for the map (East -> E, Street -> St, ...).
    """
    for pat, short in ABBREV:
        name = re.sub(pat, short, name)
    return name


def map_figure(area, streets, title, subtitle, crs, scale_bar="metres"):
    """!
    @brief A map: faded OpenStreetMap background with blocks between the study streets, street
           centrelines and names, lon/lat axes with a light grid, a scale bar and a north arrow.
           Drawn in web Mercator (the tiles' own projection, so they line up exactly); ticks in degrees.
    @param area      GeoSeries with the study area polygon (any CRS).
    @param streets   GeoDataFrame of street lines with `name` (study-streets.gpkg; any CRS).
    @param title     Bold first line.
    @param subtitle  Bold second line.
    @param crs       The metre CRS the blocks and the scale bar are measured in (the config's projected_crs).
    @param scale_bar "metres", or "feet" (a round length in feet with its metres): the config's map_style.scale_bar.
    @return (fig, ax); plot data on `ax` after .to_crs(3857), then save_map().
    """
    import contextily as ctx
    import shapely
    from geopandas import GeoSeries
    from matplotlib.ticker import FixedLocator, MaxNLocator

    R = 6378137.0                                          # web Mercator sphere radius
    to_x = lambda lon: np.radians(lon) * R
    to_y = lambda lat: R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    to_lon = lambda x: np.degrees(x / R)
    to_lat = lambda y: np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)

    area_m = area.to_crs(crs).union_all()
    st_m = streets.to_crs(crs)
    w, s, e, n = area.to_crs(3857).total_bounds
    pad = 0.18 * max(e - w, n - s)
    xlim, ylim = (w - pad, e + pad), (s - pad, n + pad)
    k = 1 / np.cos(np.radians(to_lat((s + n) / 2)))       # web-Mercator units per real metre here

    fig, ax = plt.subplots(figsize=(10, 8.6))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    # background: OSM tiles (with the identifying User-Agent OSM's tile policy requires), greyed and faded
    img, ext = ctx.bounds2img(*xlim[:1], ylim[0], xlim[1], ylim[1], source=ctx.providers.OpenStreetMap.Mapnik,
                              headers={"User-Agent": "street-interface-matrix/1.0 (research)"})
    grey = img[..., :3] @ [0.299, 0.587, 0.114]
    ax.imshow(255 - (255 - grey) * 0.3, extent=ext, cmap="gray", vmin=0, vmax=255, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, ls="--", lw=0.5, color="#c9ced6")
    ax.text(0.006, 0.006, "© OpenStreetMap contributors", transform=ax.transAxes, fontsize=6, color=MUTED)

    # blocks: the faces between streets, inset so the streets read as gaps
    lines = shapely.get_parts(shapely.union_all(st_m.geometry.values))
    between_streets = [f for f in shapely.get_parts(shapely.polygonize(lines))
                       if area_m.buffer(10).contains(f.representative_point())]
    blocks = GeoSeries([f.buffer(-6) for f in between_streets], crs=crs)
    blocks[~blocks.is_empty].to_crs(3857).plot(ax=ax, color="#e3e6ea", alpha=0.55, edgecolor="#9aa3ad",
                                               lw=0.6, zorder=1)
    st_m.to_crs(3857).plot(ax=ax, color="#8a9099", lw=0.6, ls="-.", zorder=1)

    # street names at the west end (east-west streets) or north end (north-south), along the street
    diag = np.hypot(*np.subtract(area_m.bounds[2:], area_m.bounds[:2]))
    for name, g in st_m.groupby("name"):
        merged = shapely.line_merge(shapely.union_all(g.geometry.values))
        line = max(shapely.get_parts(merged), key=lambda l: l.length)
        if line.length < 0.25 * diag:                     # only the longer streets: keeps dense areas legible
            continue
        (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
        ew = abs(x1 - x0) > abs(y1 - y0)
        if (ew and x1 > x0) or (not ew and y1 < y0):      # orient the line so its last point is the labelled end
            (x0, y0), (x1, y1) = (x1, y1), (x0, y0)
        ux, uy = (x1 - x0) / np.hypot(x1 - x0, y1 - y0), (y1 - y0) / np.hypot(x1 - x0, y1 - y0)
        at = GeoSeries.from_xy([x1 + 12 * ux], [y1 + 12 * uy], crs=crs).to_crs(3857).iloc[0]
        angle = np.degrees(np.arctan2(uy, ux))            # the map keeps true proportions
        angle = angle - 180 if angle > 90 else angle + 180 if angle < -90 else angle
        ax.text(at.x, at.y, _label(name), rotation=angle, rotation_mode="anchor", fontsize=7.5,
                fontweight="bold", color=NAVY, ha="right" if ew else "left", va="center", zorder=5,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.6})

    # scale bar: a round length (in feet with its metres, or in metres), end ticks, lower left
    width_m = (xlim[1] - xlim[0]) / k
    unit = 0.3048 if scale_bar == "feet" else 1.0
    n = max(f for f in (50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000) if f * unit <= 0.22 * width_m)
    label = f"{n:,} FT\n({n * unit:,.1f} M)" if scale_bar == "feet" else f"{n:,} M"
    x0, y0 = xlim[0] + 0.05 * (xlim[1] - xlim[0]), ylim[0] + 0.1 * (ylim[1] - ylim[0])
    x1 = x0 + n * unit * k
    tick = 0.012 * (ylim[1] - ylim[0])
    ax.plot([x0, x1], [y0, y0], color=NAVY, lw=2, solid_capstyle="butt", zorder=6)
    for x in (x0, x1):
        ax.plot([x, x], [y0 - tick, y0 + tick], color=NAVY, lw=1.5, zorder=6)
    ax.text((x0 + x1) / 2, y0 - 2.5 * tick, label, ha="center", va="top",
            fontsize=7.5, fontweight="bold", color=NAVY, zorder=6)

    # north arrow, upper right
    ax.annotate("N", xy=(0.95, 0.96), xytext=(0.95, 0.87), xycoords="axes fraction", ha="center", va="top",
                fontsize=9, fontweight="bold", color="#c0392b",
                arrowprops={"arrowstyle": "-|>", "color": "#c0392b", "lw": 1.6})

    # degree ticks on the web-Mercator axes
    ax.xaxis.set_major_locator(FixedLocator(to_x(MaxNLocator(6).tick_values(to_lon(xlim[0]), to_lon(xlim[1])))))
    ax.yaxis.set_major_locator(FixedLocator(to_y(MaxNLocator(6).tick_values(to_lat(ylim[0]), to_lat(ylim[1])))))
    ax.xaxis.set_major_formatter(lambda x, _: f"{to_lon(x):.3f}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{to_lat(y):.3f}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Longitude (degrees)", fontsize=9, color=NAVY)
    ax.set_ylabel("Latitude (degrees north)", fontsize=9, color=NAVY)
    ax.tick_params(labelsize=8, colors=NAVY)
    for spine in ax.spines.values():
        spine.set_color(NAVY)
    return fig, ax


def save_map(fig, ax, path):
    """!
    @brief Add the legend and write the map.
    @param fig, ax  From map_figure(), with labelled artists plotted.
    @param path     Output .png path, in the calling stage's folder.
    """
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95, edgecolor="#cfd5dc",
              labelcolor=NAVY, markerscale=1.4)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figure: {path}")


# ---------------------------------------------------------------- 3D buildings
COLUMN_M = 7                  # a node column's side
COLUMN_MAX_M = 220            # the busiest node's column height


def rings(geom):
    """!
    @brief The exterior rings of a Polygon or MultiPolygon, as (n, 2) arrays without the closing point.
    """
    parts = getattr(geom, "geoms", [geom])
    return [np.asarray(p.exterior.coords)[:-1, :2] for p in parts]


def prism(ring, h):
    """!
    @brief The roof and walls of a footprint ring extruded from 0 to h: a list of (k, 3) faces, roof first.
    """
    top = np.c_[ring, np.full(len(ring), h)]
    nxt = np.roll(ring, -1, axis=0)
    walls = [np.array([[*a, 0], [*b, 0], [*b, h], [*a, h]]) for a, b in zip(ring, nxt)]
    return [top, *walls]


def faces(buildings, origin, colour_of):
    """!
    @brief Every face of every building, its colour (the roof as given, walls darker), shifted to `origin`.
    """
    F, C = [], []
    for geom, h in zip(buildings.geometry, buildings.height_m):
        c = np.asarray(colour_of(h))
        for ring in rings(geom):
            f = prism(ring - origin, h)
            F += f
            C += [c] + [np.r_[c[:3] * 0.78, c[3]]] * (len(f) - 1)
    return F, C


def draw(F, C, extent, origin, title, path, cbar=None):
    """!
    @brief Draw the faces `F` in colours `C` in 3D, in true proportions over `extent` (x0, x1, y0, y1), and save.
    @param cbar  (ScalarMappable, label) for a colour bar, or None.
    """
    fig = plt.figure(figsize=(DOUBLE_IN, 5.2))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.add_collection3d(Poly3DCollection(F, facecolors=C, edgecolors=(0, 0, 0, 0.1), linewidths=0.08), autolim=False)
    x0, x1, y0, y1 = np.subtract(extent, [origin[0], origin[0], origin[1], origin[1]])
    zmax = max(f[:, 2].max() for f in F)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((x1 - x0, y1 - y0, zmax), zoom=1.2)     # true proportions, filling the frame
    ax.view_init(elev=32, azim=-62)
    ax.set_axis_off()
    ax.set_title(title, fontsize=7.5)
    if cbar:
        cb = fig.colorbar(cbar[0], ax=ax, shrink=0.4, aspect=20, pad=0.0)
        cb.set_label(cbar[1], fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)


def clip(buildings, extent):
    """!
    @brief The buildings within `extent` (x0, x1, y0, y1).
    """
    x0, x1, y0, y1 = extent
    return buildings.cx[x0:x1, y0:y1]


def png_by_height(buildings, extent, path):
    """!
    @brief The buildings extruded to their roof height, coloured by it (log scale).
    """
    origin = np.array([extent[0], extent[2]])
    norm = LogNorm(5, buildings.height_m.max())
    cmap = plt.get_cmap("YlOrBr")
    F, C = faces(buildings, origin, lambda h: cmap(norm(h)))
    sm = plt.cm.ScalarMappable(norm, cmap)
    draw(F, C, extent, origin, f"Buildings extruded to roof height ({len(buildings):,} footprints)", path,
         (sm, "roof height (m)"))


def png_footfall_columns(buildings, nf, extent, path, cmap):
    """!
    @brief The buildings in grey and a column at every node: height = its pedestrian volume, colour = M.
    """
    origin = np.array([extent[0], extent[2]])
    F, C = faces(buildings, origin, lambda h: (0.86, 0.85, 0.83, 0.55))
    d = nf[nf.M.notna() & nf.volume.notna()]
    norm = Normalize(d.M.min(), d.M.max())
    scale = COLUMN_MAX_M / d.volume.max()
    half = COLUMN_M / 2
    for x, y, v, m in zip(d.geometry.x - origin[0], d.geometry.y - origin[1], d.volume, d.M):
        ring = np.array([[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]])
        f = prism(ring, max(v * scale, 1))
        c = np.asarray(cmap(norm(m)))
        F += f
        C += [c] + [np.r_[c[:3] * 0.8, 1]] * (len(f) - 1)
    sm = plt.cm.ScalarMappable(norm, cmap)
    title = f"A column per node: height = pedestrian volume (tallest {d.volume.max():,.0f} people/h), colour = M"
    draw(F, C, extent, origin, title, path, (sm, "M (1-7)"))


def buildings_3d_figures(paths, nf, base, cmap):
    """!
    @brief The 3D figures, when the run has footprints.
    @param nf, base  paper_figures.all_maps()'s node frame and basemap (its footprints and extent).
    """
    if base.buildings is None:
        return
    b = clip(base.buildings, base.extent)
    png_by_height(b, base.extent, paths.figures / "3d-buildings-by-height.png")
    if "M" in nf and "volume" in nf:
        png_footfall_columns(b, nf, base.extent, paths.figures / "3d-footfall-columns-coloured-by-m.png", cmap)


# ---------------------------------------------------------------- the nodes map (sample_street_nodes, make_figures)
def draw_nodes_map(cfg, nodes, area_ll, streets, excluded, path):
    """!
    @brief Map of the nodes: nodes by street type, the excluded ones crossed out. Step sample_street_nodes draws it
           as a check before any imagery is bought (excluded: the review's exclusions), step make_figures for the
           paper (excluded: every node unusable by then).
    @param nodes     The nodes (nodes.gpkg), with typology.
    @param area_ll   Study area polygon, EPSG:4326.
    @param streets   Study street lines with `name` (study-streets.gpkg).
    @param excluded  The node ids to cross out.
    @param path      The .png to write.
    """
    F = cfg["sample_osm_nodes"]
    ll = nodes.to_crs(3857)                               # the map's projection
    fig, ax = map_figure(gpd.GeoSeries([area_ll], crs=4326), streets,
                         f"Sampling nodes — {cfg['name']}",
                         f"{len(ll):,} nodes on {ll.street.nunique():,} streets, every {F['spacing_m']} m "
                         f"· OSM as of {cfg.get('osm_date', 'today')[:10]}", int(cfg["projected_crs"]),
                         (cfg.get("map_style") or {}).get("scale_bar", "metres"))
    gpd.GeoSeries([area_ll.boundary], crs=4326).to_crs(3857).plot(ax=ax, color=MUTED, lw=0.9, ls="--",
                                                                  label="study area")
    for label, color in typology_colors(cfg).items():
        sub = ll[ll.typology == label]
        if len(sub):
            ax.scatter(sub.geometry.x, sub.geometry.y, s=11, c=color, edgecolors="white", linewidths=0.4,
                       zorder=3, label=f"{label.replace('_', ' ') if F.get('typology') else 'nodes'} ({len(sub):,})")
    out = ll[ll.node_id.isin(excluded)]
    if len(out):
        ax.scatter(out.geometry.x, out.geometry.y, s=34, marker="x", c=INK, linewidths=1.1, zorder=4,
                   label=f"excluded ({len(out)})")
    save_map(fig, ax, path)
