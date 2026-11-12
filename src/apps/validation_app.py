"""!
@package validation_app
@brief Rating validation, interactive: does each VLM rating follow what the photo shows? Term by term, then view by
       view, for one experiment; and the results of a fast_inference run. Not a pipeline step (docs/apps.md):

    uv run streamlit run src/apps/validation_app.py

The sidebar picks the experiment and which views to count; two tabs:
  Ratings check            every rated term in plain words: what it is checked against (its share of the
                           segmented view over the view's own arc, as validate_vlm_ratings reads it, or its record),
                           how far the two agree, whether the rating repeats on the next view and how bunched it is,
                           and a verdict. Click a term for its definition, its rating against its counterpart and the
                           views where the two disagree most; click a view to see it, its frames' segmentation and its
                           rung probabilities.
  Fast inference results   a fast_inference output folder: the class shares, each image's label map over it
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # src/: streamlit runs this file from its own folder
from analysis.vlm_rating_validation import ANCHOR, rating_repeatability
from common.data_readers import IGNORE, TermSharesByBearing, arc_shares, label_map_path, rated_fields
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
                "see it; the first is shown until you do.")
    c = st.columns([2, 3])
    shown = v.assign(view=v.file.str.rsplit("/", n=1).str[-1]).rename(columns={field + "_ev": "rating", col: label})
    pick = c[0].dataframe(shown[["view", "rating", label, "disagreement"]].round(2), hide_index=True, width="stretch",
                          on_select="rerun", selection_mode="single-row", key=f"views_{field}_{col}", height=420)
    c[0].download_button("download all views (csv)", d[["file", "node_id", field + "_ev", col]].to_csv(index=False),
                         f"{field}-vs-{col}.csv")
    with c[1]:
        view(paths, d, field, v.file.iloc[pick.selection.rows[0] if pick.selection.rows else 0])


def view(paths, d, field, file):
    """!
    @brief One view: the 180-degree view, the term's rung probabilities, its node's frames over their label maps.
    """
    r = d.set_index("file").loc[file]
    st.image(str(paths.views / file), caption=file, width="stretch")
    ps = [f"{field}_p{k}" for k in range(1, 8)]
    if all(p in r for p in ps):
        st.bar_chart(pd.Series(r[ps].to_numpy(float), index=range(1, 8), name="probability"), x_label="rung",
                     height=160)
    frames = pd.read_csv(paths.raw_photos_csv)
    frames = frames[frames.node_id == r.node_id].sort_values("heading")
    for col, f in zip(st.columns(max(len(frames), 1)), frames.itertuples()):
        lab = label_map_path(paths, f.node_id, f.heading)
        col.image(overlay(paths.raw_photos / f.file, lab) if lab.exists() else str(paths.raw_photos / f.file),
                  caption=f"{f.heading:g}° segmented", width="stretch")


def fast_inference_results():
    """!
    @brief A fast_inference output folder: its class shares, and each image's label map over it.
    """
    out = Path(st.text_input("fast_inference output folder (it holds class-shares.csv)", "")).expanduser()
    if not (out / "class-shares.csv").exists():
        st.info("Give the output folder of a fast_inference run: src/apps/fast_inference.sh <image folder> writes "
                "<image folder>-segmented/.")
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
    st.set_page_config(page_title="Rating validation", layout="wide")
    sb = st.sidebar
    sb.title("Rating validation")
    names = sorted(p.stem for p in EXPERIMENTS.glob("*.yaml"))
    exp = sb.selectbox("Experiment", names, index=names.index(CONFIG.stem))
    usable = sb.toggle("usable nodes only", value=True)
    ratings, inference = st.tabs(["Ratings check", "Fast inference results"])
    with inference:
        fast_inference_results()
    if not Paths(REPO, load_config(EXPERIMENTS / f"{exp}.yaml")).ratings_csv.exists():
        ratings.info(f"{exp} has no ratings yet: run pipeline step rate_views_with_vlm for it.")
        return
    cfg, d = load(exp)
    d = d[d.usable.astype(bool)] if usable else d
    street = sb.selectbox("Street", ["all streets", *sorted(d.street.unique())])
    d = d if street == "all streets" else d[d.street == street]
    sb.caption(f"{len(d)} views on {d.node_id.nunique()} nodes")
    with ratings:
        ratings_check(cfg, Paths(REPO, cfg), d)


if __name__ == "__main__":                                          # streamlit runs it; fast_inference imports overlay
    main()
