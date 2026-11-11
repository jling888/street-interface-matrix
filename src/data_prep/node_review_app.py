"""!
@package node_review_app
@brief Node review: look at each node's imagery and record ok / exclude / keep.

Interactive, so not a pipeline step (the pipeline never runs it):

    streamlit run src/data_prep/node_review_app.py

Shows, one node at a time in walking order: the four Street View frames (download_street_view), their
segmentation (fuse_label_maps, on a toggle), the two 180-degree views if stitch_180_views has run, the
panorama's date / photographer / distance, and the automatic verdict from read_nodes(). Every click is saved at
once to data/<city>/data_prep/node-review-decisions.csv, which read_nodes() applies on the next read
by any step.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # src/: streamlit runs this file from its own folder
from common.data_readers import IGNORE, class_taxonomy, label_map_path
from common.map_style import SLOTS
from common.node_table import node_review, read_nodes
from common.settings_and_paths import CONFIG, REPO, Paths, load_config

FILTERS = ["all", "not reviewed yet", "flagged by the pipeline", "user uploads", "re-probed (panorama off the node)",
           "excluded by review"]


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


def read(path):
    return pd.read_csv(path) if path.exists() else None


def save(paths, nid, decision, reason):
    """!
    @brief Record one decision, replacing any earlier one for the node.
    @param nid       Node id.
    @param decision  "ok", "exclude" or "keep".
    @param reason    Free text (kept for exclude / keep).
    """
    review = node_review(paths)
    row = pd.DataFrame([{"node_id": nid, "decision": decision, "reason": reason if decision != "ok" else ""}])
    review = pd.concat([review[review.node_id != nid], row]).sort_values("node_id")
    review.to_csv(paths.node_review, index=False)


def step(order, by):
    """Move the current node by `by` within `order` (a button callback, so it runs before the rerun)."""
    i = order.index(st.session_state.node) if st.session_state.get("node") in order else 0
    st.session_state.node = order[max(0, min(len(order) - 1, i + by))]


def decide(paths, order, nid, decision):
    save(paths, nid, decision, st.session_state.get(f"reason_{nid}", "").strip())
    step(order, 1)


def mini_map(nodes, nid):
    fig, ax = plt.subplots(figsize=(3, 3))
    street = nodes.loc[nodes.node_id == nid, "street"].iloc[0]
    ax.scatter(nodes.lon, nodes.lat, s=2, c="#c9ced6")
    on = nodes[nodes.street == street]
    ax.scatter(on.lon, on.lat, s=4, c=SLOTS[0])
    me = nodes[nodes.node_id == nid]
    ax.scatter(me.lon, me.lat, s=40, c="#c0392b", zorder=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


def main(cfg, paths):
    st.set_page_config(page_title=f"Node review — {cfg['name']}", layout="wide")
    name = {i: c for c, i in class_taxonomy(cfg).items()}
    nodes = read_nodes(paths).sort_values(["street", "seq_fwd"]).reset_index(drop=True)
    meta, manifest, renders = (read(p) for p in (paths.pano_metadata_csv, paths.raw_photos_csv, paths.views_csv))
    review = node_review(paths).set_index("node_id")
    if manifest is None:
        st.error("No imagery yet: run pipeline step download_street_view first.")
        return
    meta = meta.set_index("node_id") if meta is not None else pd.DataFrame()

    # ---- sidebar: which nodes to go through
    sb = st.sidebar
    sb.title("Node review")
    sb.caption(cfg["name"])
    street = sb.selectbox("Street", ["all streets", *sorted(nodes.street.unique())])
    show = sb.radio("Show", FILTERS)
    pick = nodes if street == "all streets" else nodes[nodes.street == street]
    if show == "not reviewed yet":
        pick = pick[~pick.node_id.isin(review.index)]
    elif show == "flagged by the pipeline":
        pick = pick[pick.exclude_reason != ""]
    elif show == "user uploads" and "user_upload" in meta:
        pick = pick[pick.node_id.map(meta.user_upload).fillna(False).astype(bool)]
    elif show == "re-probed (panorama off the node)" and "probe_shift_m" in meta:   # a user upload swapped for Google's
        pick = pick[pick.node_id.map(meta.probe_shift_m).fillna(0).ne(0)]
    elif show == "excluded by review":
        pick = pick[pick.node_id.isin(review.index[review.decision == "exclude"])]
    order = pick.node_id.tolist()
    sb.progress(len(review) / len(nodes), text=f"reviewed {len(review)} of {len(nodes)}")
    if not order:
        st.info("No nodes match this filter.")
        return
    if st.session_state.get("node") not in order:
        st.session_state.node = order[0]
    sb.selectbox("Node", order, key="node")
    nid = st.session_state.node
    sb.pyplot(mini_map(nodes, nid))

    # ---- the node: facts, verdicts, imagery
    node = nodes.set_index("node_id").loc[nid]
    m = meta.loc[nid] if nid in meta.index else None
    st.subheader(f"{nid}  ·  {order.index(nid) + 1} of {len(order)}")
    facts = [node.osm_name, node.typology.replace("_", " ")]
    if m is not None:
        facts += [f"captured {m.pano_date}", str(m.copyright), f"{m.pano_offset_m:.1f} m from node"]
    st.caption("  ·  ".join(facts))
    verdict = "usable" if node.usable else f"unusable: {node.exclude_reason}"
    mine = f" — review: {review.decision[nid]}" + (f" ({review.reason[nid]})" if review.reason[nid] else "") \
        if nid in review.index else " — not reviewed yet"
    (st.success if node.usable else st.warning)(verdict + mine)

    frames = manifest[manifest.node_id == nid].sort_values("heading")
    for col, f in zip(st.columns(max(len(frames), 1)), frames.itertuples()):
        col.image(str(paths.raw_photos / f.file), caption=f"{f.heading:g}°", width="stretch")
    labels = [label_map_path(paths, nid, f.heading) for f in frames.itertuples()]
    if any(p.exists() for p in labels) and st.toggle("show segmentation (fused label map)", value=False):
        for col, f, p in zip(st.columns(max(len(frames), 1)), frames.itertuples(), labels):
            if p.exists():
                col.image(overlay(paths.raw_photos / f.file, p), caption=f"{f.heading:g}° segmented", width="stretch")
        shown = sorted({int(i) for p in labels if p.exists() for i in np.unique(np.asarray(Image.open(p)))} - {255})
        st.markdown("  ".join(f"<span style='color:#%02x%02x%02x'>■</span> {name[i]}" % color(i)
                              for i in shown), unsafe_allow_html=True)
    if renders is not None:
        views = renders[renders.node_id == nid]
        for col, v in zip(st.columns(max(len(views), 1)), views.itertuples()):
            col.image(str(paths.views / v.file), caption=f"180° view, {v.walk.replace('_', ' ')}",
                      width="stretch")

    # ---- decision; each click saves and moves to the next node
    st.text_input("Reason (for exclude / keep)", value=review.reason.get(nid, "") if nid in review.index else "",
                  key=f"reason_{nid}")
    c = st.columns([1, 1, 1, 3, 1, 1])
    c[0].button("ok", on_click=decide, args=(paths, order, nid, "ok"), width="stretch", type="primary")
    c[1].button("exclude", on_click=decide, args=(paths, order, nid, "exclude"), width="stretch")
    c[2].button("keep", on_click=decide, args=(paths, order, nid, "keep"), width="stretch",
                help="use this node even though the pipeline flagged it")
    c[4].button("◀ prev", on_click=step, args=(order, -1), width="stretch")
    c[5].button("next ▶", on_click=step, args=(order, 1), width="stretch")


cfg = load_config(CONFIG)
main(cfg, Paths(REPO, cfg))
