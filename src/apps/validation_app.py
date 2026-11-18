"""!
@package validation_app
@brief Rating validation, interactive: does each VLM rating follow what the photo shows? Term by term, then view by
       view, for one experiment; and the results of a fast_inference run. Not a pipeline step (docs/apps.md):

    uv run streamlit run src/apps/validation_app.py
    uv run streamlit run src/apps/validation_app.py -- --output_path <fast_inference output folder>

The sidebar picks the experiment and which views to count; two views, chosen at the top:
  Walk    one street, view by view, as a pedestrian walks it (the manner of the earlier build_walk_interface page):
          the 180-degree view, a strip of M along the walk, every rating grouped by the metric's pillars beside what
          was measured for the same view (the segmented share or the record) and whether the two agree, M and its
          pillars, and where the view sits among all views. Step, play, or turn round.
  Terms   every rated term in plain words: what it is checked against (its share of the segmented view over the
          view's own arc, as validate_vlm_ratings reads it, or its record), how far the two agree, whether the rating
          repeats on the next view and how bunched it is, and a verdict. Click a term for its definition, its rating
          against its counterpart and the views where the two disagree most; click a view to walk to it.
With --output_path, the page shows that fast_inference output folder instead: the class shares, each image's label
map over it.
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # src/: streamlit runs this file from its own folder
from analysis.vlm_rating_validation import ANCHOR, rating_repeatability
from common.data_readers import (IGNORE, TermSharesByBearing, arc_shares, indicator_ratings, label_map_path,
                                 rated_fields)
from common.node_table import node_table
from common.settings_and_paths import CONFIG, CONFIGS, REPO, Paths, load_config

EXPERIMENTS = CONFIGS / "experiments"
# each segmented counterpart (ANCHOR) in words
SEGMENTED = {"V_nat": "vegetation above ground", "vegetation_eye": "vegetation at eye level",
             "green_share_eye": "vegetation / (vegetation + walls), eye level", "V_built": "buildings",
             "enclosure": "1 - sky", "signage": "signboards", "V_pave": "sidewalk",
             "openings_eye": "windows and doors at eye level", "seating": "benches and stoops"}
# the verdict on a term's rank agreement with its counterpart (a reading, not a test: no benchmark exists)
FOLLOWS, WEAK = 0.5, 0.3
# one view: how far apart its rating and its counterpart sit among all views (percentile ranks, 0-1)
AGREES, DISAGREES = 0.25, 0.5
PILLAR_COLORS = ["#4c9a62", "#d08a4c", "#4a86b0", "#8a8a8a"]


def color(i):
    """!
    @brief Display colour of taxonomy id `i`: a fixed qualitative palette; unknown is grey.
    @return (r, g, b) in 0-255.
    """
    if i == 0:
        return (128, 128, 128)
    rgb = matplotlib.colormaps["tab20"]((i - 1) % 20) if i <= 20 else matplotlib.colormaps["Dark2"]((i - 21) % 8)
    return tuple(int(255 * v) for v in rgb[:3])


def overlay(photo, label):
    """!
    @brief A photo with its fuse label map blended over it, in the taxonomy's colours (ignored pixels black).
    @return HxWx3 uint8 image.
    """
    lab = np.asarray(Image.open(label))
    pal = np.array([color(i) if i != IGNORE else (0, 0, 0) for i in range(256)], float)
    return (0.45 * np.asarray(Image.open(photo).convert("RGB"), float) + 0.55 * pal[lab]).astype(np.uint8)


@st.cache_data
def load(exp):
    """!
    @brief An experiment's rated views with what they are checked against: seg_<counterpart>, each term's share over
           the view's own arc (the facade frames' arcs for a term rated on them, as validate_vlm_ratings reads it), and
           the node's records and usable flag.
    @return (cfg, the views).
    """
    cfg = load_config(EXPERIMENTS / f"{exp}.yaml")
    paths = Paths(REPO, cfg)
    d = pd.read_csv(paths.ratings_csv)
    nodes = pd.DataFrame(node_table(cfg, paths).drop(columns="geometry"))
    d = d.merge(nodes[[c for c in nodes if c not in d or c == "node_id"]], on="node_id", how="left")
    if paths.term_shares_by_bearing_npz.exists():
        args = (dict(np.load(paths.term_shares_by_bearing_npz)), pd.read_csv(paths.views_csv).set_index("file").bearing,
                TermSharesByBearing(cfg))
        along = arc_shares(d, *args, cfg["stitch_180_views"]["fov"])
        fov = cfg["download_gsv_panos"]["fov"]
        sides = (arc_shares(d, args[0], args[1] - 90, args[2], fov) + arc_shares(d, args[0], args[1] + 90, args[2], fov)) / 2
        d = d.join(pd.DataFrame({"seg_" + c: (sides if f + "_left_rating" in d else along)[c]
                                 for f, c in ANCHOR.items() if f + "_ev" in d}))
    return cfg, d


def counterparts(cfg, d, field):
    """!
    @return [(what in words, its column in d, the sign expected)]: the term's segmented counterpart, then its record.
    """
    agree = cfg["vlm_rating_validation"]["agreement"]
    out = [(f"segmented {SEGMENTED.get(ANCHOR[field], ANCHOR[field])}", "seg_" + ANCHOR[field], 1)] \
        if field in ANCHOR and "seg_" + ANCHOR[field] in d else []
    if field in agree and agree[field]["record"] in d:
        out.append((f"record: {agree[field]['record'].replace('_', ' ')}", agree[field]["record"], agree[field]["sign"]))
    return out


def agreement(d, field, col, sign):
    """!
    @return (Spearman rho turned by the sign expected, so higher is better; p; views), NaN with under three views.
    """
    v = d[[field + "_ev", col]].dropna()
    if len(v) < 3:
        return np.nan, np.nan, len(v)
    rho, p = spearmanr(v[field + "_ev"], v[col])
    return sign * rho, p, len(v)


def follows(rho):
    """!
    @brief A term's agreement with its counterpart, in words.
    """
    return ("no counterpart to check against" if np.isnan(rho) else "✓ follows the photo" if rho >= FOLLOWS
            else "~ weak" if rho >= WEAK else "✗ does not follow")


# ---------------------------------------------------------------- the walk
def pillars(cfg, fields):
    """!
    @return {pillar name: [rated terms]}: the metric's pillars in its order, each with the terms its indicators read
            (a derived rating as the two it is made from), then the terms the score does not use.
    """
    C, out = cfg["metric"], {}
    derived = C.get("derived_ratings") or {}
    for k, items in C["dimensions"].items():
        rs = [x for it in items for r in indicator_ratings(it) for x in derived.get(r, [r]) if x in fields]
        out[C["names"][k]] = list(dict.fromkeys(rs))
    rest = [f for f in fields if not any(f in v for v in out.values())]
    return out | ({"Not in the score": rest} if rest else {})


def short(cfg, f):
    """A rated term's name in words, from its definition."""
    return cfg["vlm"]["prompts"]["terms"].get(f, {}).get("definition", f).split(":")[0]


def measured(cfg, d, f, r):
    """!
    @brief What was measured for one view's term, and whether its rating agrees: how far apart the two sit among all
           views (percentile ranks, the record turned by its expected sign).
    @return A line of text; "" when nothing measures the term.
    """
    against = counterparts(cfg, d, f)
    if not against or pd.isna(r.get(against[0][1])) or pd.isna(r[f + "_ev"]):
        return ""
    label, col, sign = against[0]
    v = r[col]
    gap = abs((d[f + "_ev"] <= r[f + "_ev"]).mean() - ((sign * d[col]) <= sign * v).mean())
    mark = "✓ agrees" if gap < AGREES else "⚠ disagrees" if gap >= DISAGREES else "~ apart"
    return f"{label}: {f'{v:.0%}' if col.startswith('seg_') else f'{v:,.3g}'} · {mark}"


def bars(rows, color, lo=1, hi=7):
    """!
    @brief Labelled bars on a lo-hi scale, each with a line of small text under it, as HTML.
    @param rows  [(label, value, note)].
    """
    out = []
    for label, v, note in rows:
        w = 0 if pd.isna(v) else 100 * (v - lo) / (hi - lo)
        out.append(f"<div style='display:flex;align-items:center;gap:8px;font-size:0.85rem'>"
                   f"<div style='width:42%'>{label}</div><div style='flex:1;background:rgba(128,128,128,.18);"
                   f"height:10px;border-radius:2px'><div style='width:{w:.0f}%;background:{color};height:10px;"
                   f"border-radius:2px'></div></div><div style='width:2.4em;text-align:right'>"
                   f"{'-' if pd.isna(v) else f'{v:.1f}'}</div></div>"
                   f"<div style='font-size:0.72rem;opacity:.65;margin:0 0 6px 42%'>{note}</div>")
    return "".join(out)


@st.cache_data
def strip(values, at):
    """!
    @brief M along the walk as a coloured strip (unusable views grey), the current view marked.
    """
    cmap = matplotlib.colormaps["magma"].with_extremes(bad="#cccccc")
    fig, ax = plt.subplots(figsize=(8, 0.35))
    ax.imshow(np.ma.masked_invalid(np.array([values], float)), aspect="auto", cmap=cmap, interpolation="nearest")
    ax.axvline(at, color="#29b6f6", lw=2.5)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


@st.cache_data
def where(values, v):
    """!
    @brief Where a view's M sits among every view's, as a histogram with the view marked.
    """
    fig, ax = plt.subplots(figsize=(3, 1.1))
    ax.hist(values, bins=30, color="#e7a17a")
    ax.axvline(v, color="#29b6f6", lw=2)
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def walk(cfg, paths, d, scores):
    """!
    @brief One street, view by view: the street and the direction of the walk, then its views (step_view).
    """
    views = pd.read_csv(paths.views_csv)
    c = st.columns([3, 3, 1])
    street = c[0].selectbox("Street", sorted(views.street.unique()), format_func=lambda x: x.replace("_", " "),
                            key="walk_street")
    way = c[1].radio("Walking", sorted(views[views.street == street].walk.unique()), horizontal=True,
                     format_func=lambda w: w.replace("_", " "), key=f"way_{street}")
    route = views[(views.street == street) & (views.walk == way)].sort_values("seq").reset_index(drop=True)
    playing = c[2].toggle("play", key="play")
    st.fragment(step_view, run_every=1.5 if playing else None)(cfg, paths, d, scores, route, playing)


def step_view(cfg, paths, d, scores, route, playing):
    """!
    @brief The walk's current view: the controls, the view and M's strip, the ratings beside what was measured, M and
           where it sits. A fragment, run again every 1.5 s while playing.
    """
    files = list(route.file)
    if st.session_state.get("route") != files:                  # another street or direction: start at its beginning
        st.session_state.route, st.session_state.step = files, 1
    if st.session_state.get("goto") in files:                   # a view chosen in Terms
        st.session_state.step = files.index(st.session_state.pop("goto")) + 1
    elif playing:
        st.session_state.step = st.session_state.step % len(files) + 1
    move = lambda by: st.session_state.update(step=min(max(st.session_state.step + by, 1), len(files)))
    c = st.columns([1, 1, 8])
    c[0].button("◀", on_click=move, args=(-1,), width="stretch")
    c[1].button("▶", on_click=move, args=(1,), width="stretch")
    c[2].slider("view", 1, max(len(files), 2), key="step", label_visibility="collapsed")
    i = st.session_state.step - 1
    file, node = files[i], route.node_id.iloc[i]
    r = d.set_index("file").loc[file] if file in set(d.file) else None
    s = scores.set_index("file").loc[file] if scores is not None and file in set(scores.file) else None

    left, right = st.columns([3, 2])
    with left:
        st.caption(f"{node}  ·  view {i + 1} of {len(files)}  ·  facing {route.cardinal.iloc[i]}")
        st.image(str(paths.views / file), width="stretch")
        if scores is not None:
            m = route.file.map(scores.set_index("file").M)
            st.pyplot(strip(tuple(m), i))
            st.caption(f"M along this walk, {m.min():.2f} (dark) to {m.max():.2f} (light); grey: unusable views")
        with st.expander("this node's four frames, segmented"):
            frames = pd.read_csv(paths.raw_photos_csv)
            frames = frames[frames.node_id == node].sort_values("heading")
            for col, f in zip(st.columns(max(len(frames), 1)), frames.itertuples()):
                lab = label_map_path(paths, f.node_id, f.heading)
                col.image(overlay(paths.raw_photos / f.file, lab) if lab.exists() else str(paths.raw_photos / f.file),
                          caption=f"{f.heading:g}°", width="stretch")
    with right:
        if r is None:
            st.warning("This view is not rated, or its node is unusable (the sidebar can show unusable nodes).")
            return
        st.markdown("**Ratings, 1 to 7**, and what was measured for this view")
        box = st.container(height=560)                          # scrolls, so M below stays in view
        for (pillar, fs), color in zip(pillars(cfg, rated_fields(d)).items(), PILLAR_COLORS):
            box.markdown(f"<div style='font-size:0.75rem;letter-spacing:.08em;color:{color};margin-top:6px'>"
                        f"■ {pillar.upper()}</div>" + bars([(short(cfg, f), r[f + "_ev"], measured(cfg, d, f, r))
                                                           for f in fs], color), unsafe_allow_html=True)
    if s is None:
        st.info("No score for this view: run pipeline step compose_metric for this experiment.")
        return
    dims = list(cfg["metric"]["dimensions"])
    c = st.columns([1, 2, 2])
    c[0].metric("composite M (1-7)", f"{s.M:.2f}")
    c[1].markdown(bars([(cfg["metric"]["names"][k], s[k], "") for k in dims], "#7e57c2"), unsafe_allow_html=True)
    c[2].pyplot(where(tuple(scores.M), s.M))
    c[2].caption(f"where this view sits: {(scores.M <= s.M).mean():.0%} of views score at or below it")


# ---------------------------------------------------------------- the tabs
def ratings_check(cfg, paths, d):
    """!
    @brief Every rated term in plain words; a click opens one.
    """
    st.markdown("**Does each VLM rating follow what the photo shows?** Each term's rating is set against what the "
                "segmentation measures over the same view, or the city's record for its node.")
    st.markdown("1. Read the **verdict** of each term in the table.  \n"
                "2. Open a term: tick the box at the left of its row. Below the table, its rating is set against "
                "what it is checked against.  \n"
                "3. Look at the **views where the two disagree most**: click one to see it. Is the rating wrong, or "
                "the segmentation or the record?")
    V, terms = cfg["vlm_rating_validation"], cfg["vlm"]["prompts"]["terms"]
    rep = rating_repeatability(d).set_index("field")
    rows = []
    for f in rated_fields(d):
        against = counterparts(cfg, d, f)
        rho = agreement(d, f, against[0][1], against[0][2])[0] if against else np.nan
        r = rep.loc[f]
        flags = [w for w, bad in (("bunched", r.modal_share > V["max_modal_share"]),
                                  ("not repeatable", r.next_view_r < V["min_repeatability"])) if bad]
        rows.append({"term": terms[f]["definition"].split(":")[0] if f in terms else f, "field": f,
                     "checked against": against[0][0] if against else "-", "agreement": rho,
                     "repeats": r.next_view_r, "on one rung": r.modal_share, "verdict": follows(rho) + (f" ({', '.join(flags)})" if flags else "")})
    t = pd.DataFrame(rows)
    pick = st.dataframe(t, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="terms",
                        column_config={
                            "field": st.column_config.TextColumn(help="the term's name in the code and the tables"),
                            "agreement": st.column_config.NumberColumn(format="%.2f", help=(
                                "rank agreement (Spearman rho) of the rating with what it is checked against, turned "
                                f"so higher is better: {FOLLOWS} and above follows the photo, under {WEAK} does not")),
                            "repeats": st.column_config.NumberColumn(format="%.2f", help=(
                                "correlation with the next view along the walk, 20 m on; under "
                                f"{V['min_repeatability']} the rating does not repeat")),
                            "on one rung": st.column_config.NumberColumn(format="percent", help=(
                                f"share of views on the most common rung; over {V['max_modal_share']:.0%} the ratings "
                                "are bunched and cannot follow the photos however well they are read"))})
    # the term ticked, else the first with something to check against
    field = t.field.iloc[pick.selection.rows[0]] if pick.selection.rows else t.field[t["checked against"] != "-"].iloc[0]
    term(cfg, paths, d, field, chosen=bool(pick.selection.rows))


def term(cfg, paths, d, field, chosen=True):
    """!
    @brief One term: its definition, its rating against its counterpart, and the views where the two disagree most.
    @param chosen  False when shown before any is ticked.
    """
    spec = cfg["vlm"]["prompts"]["terms"].get(field, {})
    st.divider()
    st.subheader(f"{spec.get('definition', field).split(':')[0]}  ·  `{field}`")
    if not chosen:
        st.caption("Shown first; tick the box at the left of another term's row in the table to open it.")
    if spec:
        st.caption(spec["definition"])
        with st.expander("its seven rungs"):
            st.markdown("\n".join(f"{k}. {s}" for k, s in enumerate(spec["scale"], 1)))
    against = counterparts(cfg, d, field)
    if not against:
        st.info("Nothing measures this term: its ratings only.")
        st.bar_chart(d[field + "_ev"].round().value_counts().sort_index(), x_label="rung", y_label="views")
        return
    label, col, sign = against[[a[0] for a in against].index(
        st.radio("Checked against", [a[0] for a in against], horizontal=True, key=f"against_{field}"))]
    rho, p, n = agreement(d, field, col, sign)
    st.markdown(f"Agreement **{rho:+.2f}** (p {p:.2g}, {n} views): {follows(rho)}")
    c = st.columns(2)
    v = d[["file", "node_id", field + "_ev", col]].dropna()
    c[0].scatter_chart(v, x=col, y=field + "_ev", x_label=label, y_label="rating (1-7)", height=300)
    c[1].dataframe(v.groupby(v[field + "_ev"].round())[col].describe()[["count", "25%", "50%", "75%"]]
                   .rename_axis("rung").round(3), width="stretch")
    c[1].caption(f"{label}, per rung of the rating: the middle half of the views (25-75%)")

    # where they disagree most: the ranks furthest apart, the record's turned by its expected sign
    v["disagreement"] = (v[field + "_ev"].rank(pct=True) - (sign * v[col]).rank(pct=True)).abs()
    v = v.sort_values("disagreement", ascending=False).head(50).reset_index(drop=True)
    st.markdown("**Views where the rating and its counterpart disagree most.** Tick one at the left of its row to "
                "walk to it.")
    shown = v.assign(view=v.file.str.rsplit("/", n=1).str[-1]).rename(columns={field + "_ev": "rating", col: label})
    key = f"views_{field}_{col}"
    st.dataframe(shown[["view", "rating", label, "disagreement"]].round(2), hide_index=True, width="stretch",
                 on_select=partial(go_to_view, key, d.set_index("file").loc[v.file, ["street", "walk"]]),
                 selection_mode="single-row", key=key, height=420)
    st.download_button("download all views (csv)", d[["file", "node_id", field + "_ev", col]].to_csv(index=False),
                       f"{field}-vs-{col}.csv")


def go_to_view(key, views):
    """!
    @brief A view ticked in Terms: open Walk on its street and direction, at that view.
    @param views  The table's views (file index) with their street and walk.
    """
    if rows := st.session_state[key].selection.rows:
        r = views.iloc[rows[0]]
        st.session_state.update({"page": "Walk", "walk_street": r.street, f"way_{r.street}": r.walk, "goto": r.name})


def fast_inference_results(out):
    """!
    @brief A fast_inference output folder (--output_path): its class shares, and each image's label map over it.
    """
    st.title("Fast inference results")
    st.caption(str(out))
    if not (out / "class-shares.csv").exists():
        st.error(f"No class-shares.csv in {out}: give the output folder of a fast_inference run "
                 "(src/apps/fast_inference.sh <image folder> writes <image folder>-segmented/).")
        return
    s = pd.read_csv(out / "class-shares.csv")
    pick = st.dataframe(s.round(2), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                        key="images")
    n = s.image.iloc[pick.selection.rows[0] if pick.selection.rows else 0]
    c = st.columns([3, 1])
    c[0].image(str(out / "overlays" / f"{n}.png"), caption=n, width="stretch")
    shares = s.set_index("image").loc[n].filter(like="_pct")
    c[1].bar_chart(shares[shares > 0].rename(lambda x: x[:-4]).sort_values(ascending=False), horizontal=True)


def main():
    ap = argparse.ArgumentParser(description="Rating validation (docs/apps.md)")
    ap.add_argument("--output_path", type=Path, help="a fast_inference output folder, to look at instead")
    a = ap.parse_args()
    st.set_page_config(page_title="Rating validation", layout="wide")
    if a.output_path:
        return fast_inference_results(a.output_path.expanduser())
    sb = st.sidebar
    sb.title("Rating validation")
    names = sorted(p.stem for p in EXPERIMENTS.glob("*.yaml"))
    exp = sb.selectbox("Experiment", names, index=names.index(CONFIG.stem))
    usable = sb.toggle("usable nodes only", value=True)
    st.session_state.setdefault("page", "Walk")
    page = st.radio("Show", ["Walk", "Terms"], key="page", horizontal=True, label_visibility="collapsed")
    if not Paths(REPO, load_config(EXPERIMENTS / f"{exp}.yaml")).ratings_csv.exists():
        st.info(f"{exp} has no ratings yet: run pipeline step rate_views_with_vlm for it.")
        return
    cfg, d = load(exp)
    paths = Paths(REPO, cfg)
    d = d[d.usable.astype(bool)] if usable else d
    sb.caption(f"{len(d)} views on {d.node_id.nunique()} nodes")
    if page == "Walk":
        scores = pd.read_csv(paths.scores_csv) if paths.scores_csv.exists() else None   # the metric (compose_metric)
        walk(cfg, paths, d, scores)
    else:
        ratings_check(cfg, paths, d)


if __name__ == "__main__":                                          # streamlit runs it; fast_inference imports overlay
    main()
