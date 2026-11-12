"""!
@package node_review_app
@brief Node review: look at each node's Street View imagery and record whether it is usable: ok / exclude / keep.

Interactive, so not a pipeline step (the pipeline never runs it):

    streamlit run src/apps/node_review_app.py

Shows, one node at a time in walking order: the panorama's date / photographer / distance, the automatic verdict
from read_nodes(), the decision buttons, the four Street View frames (download_street_view), and the two 180-degree
views if stitch_180_views has run; their segmentation is checked in validation_app. Every click is saved at once to
data/<city>/data_prep/node-review-decisions.csv, which read_nodes() applies on the next read by any step, with the
panorama it was made on (pano_id): a decision on a panorama since replaced counts as not reviewed.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # src/: streamlit runs this file from its own folder
from common.map_style import SLOTS
from common.node_table import node_review, read_nodes
from common.settings_and_paths import CONFIG, REPO, Paths, load_config

# common reasons, so they can be counted; a free text may follow
REASONS = ["in tunnel", "construction or scaffolding", "view blocked", "wrong street", "indoor or off the street"]
FILTERS = ["all", "not reviewed yet", "flagged by the pipeline", "re-probed (panorama off the node)", "excluded by review"]


def read(path):
    return pd.read_csv(path) if path.exists() else None


def save(paths, nid, row):
    """!
    @brief Replace a node's decision with `row` ({node_id, decision, reason, pano_id}), or remove it (None).
    @return The decision replaced, None if the node had none (for undo).
    """
    review = node_review(paths)
    old = review[review.node_id == nid]
    review = pd.concat([review[review.node_id != nid], pd.DataFrame([row] if row else [])])
    review.sort_values("node_id").to_csv(paths.node_review, index=False)
    return old.iloc[0].to_dict() if len(old) else None


def step(order, by):
    """Move the current node by `by` within `order` (a button callback, so it runs before the rerun)."""
    i = order.index(st.session_state.node) if st.session_state.get("node") in order else 0
    st.session_state.node = order[max(0, min(len(order) - 1, i + by))]


def decide(paths, order, nid, decision, pano_id):
    """A decision button: save it with the panorama it was made on, keep what it replaced for undo, go to the next."""
    reason = "; ".join(r for r in (st.session_state.get(f"pick_{nid}"), st.session_state.get(f"reason_{nid}", "").strip())
                       if r) if decision != "ok" else ""
    old = save(paths, nid, {"node_id": nid, "decision": decision, "reason": reason, "pano_id": pano_id})
    st.session_state.undo = (nid, old)
    step(order, 1)


def undo(paths):
    """The undo button: put back what the last decision replaced, and go back to its node."""
    nid, old = st.session_state.pop("undo")
    save(paths, nid, old)
    st.session_state.node = nid


@st.cache_data(max_entries=64)
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
    nodes = read_nodes(paths).sort_values(["street", "seq_fwd"]).reset_index(drop=True)
    meta, manifest, renders = (read(p) for p in (paths.pano_metadata_csv, paths.raw_photos_csv, paths.views_csv))
    review = node_review(paths).set_index("node_id")
    if manifest is None:
        st.error("No imagery yet: run pipeline step download_street_view first.")
        return
    meta = meta.set_index("node_id") if meta is not None else pd.DataFrame()
    # a decision on a panorama since replaced (a re-download, a re-probe) counts as not reviewed
    made_on = review.pano_id if "pano_id" in review else pd.Series("", index=review.index)
    stale = set(review.index[made_on.ne("") & made_on.ne(review.index.map(meta.get("pano_id", pd.Series(dtype=str))))])

    # ---- sidebar: which nodes to go through
    sb = st.sidebar
    sb.title("Node review")
    sb.caption(cfg["name"])
    street = sb.selectbox("Street", ["all streets", *sorted(nodes.street.unique())])
    show = sb.radio("Show", FILTERS)
    pick = nodes if street == "all streets" else nodes[nodes.street == street]
    if show == "not reviewed yet":
        pick = pick[~pick.node_id.isin(review.index) | pick.node_id.isin(stale)]
    elif show == "flagged by the pipeline" and "reason" in meta:    # download's reason: no panorama, user upload, ...
        pick = pick[pick.node_id.map(meta.reason).notna()]
    elif show == "re-probed (panorama off the node)" and "probe_shift_m" in meta:   # a user upload swapped for Google's
        pick = pick[pick.node_id.map(meta.probe_shift_m).fillna(0).ne(0)]
    elif show == "excluded by review":
        pick = pick[pick.node_id.isin(review.index[review.decision == "exclude"])]
    order = pick.node_id.tolist()
    done = len(review) - len(stale)
    sb.progress(done / len(nodes), text=f"reviewed {done} of {len(nodes)}")
    sb.caption("  ·  ".join(f"{k} {n}" for k, n in review.decision.value_counts().items())
               + (f"  ·  imagery changed since review {len(stale)}" if stale else ""))
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
        if m.get("probe_shift_m"):                          # the nearest panorama was a user upload
            facts.append(f"probed {m.probe_shift_m:+g} m along the street")
    st.caption("  ·  ".join(facts))
    by_review = nid in review.index and review.decision[nid] == "exclude"   # then the review line says why
    verdict = "usable" if node.usable else "unusable" + ("" if by_review else f": {node.exclude_reason}")
    mine = f" — review: {review.decision[nid]}" + (f" ({review.reason[nid]})" if review.reason[nid] else "") \
        + (", on a panorama since replaced" if nid in stale else "") if nid in review.index else " — not reviewed yet"
    (st.success if node.usable and nid not in stale else st.warning)(verdict + mine)

    # ---- decision; each click saves and moves to the next node
    pano_id = str(m.pano_id) if m is not None else ""
    c = st.columns([2, 3])
    c[0].pills("Reason (for exclude / keep)", REASONS, key=f"pick_{nid}")
    c[1].text_input("More, or another reason", value=review.reason.get(nid, "") if nid in review.index else "",
                    key=f"reason_{nid}")
    c = st.columns([1, 1, 1, 1, 2, 1, 1])
    c[0].button("ok", on_click=decide, args=(paths, order, nid, "ok", pano_id), width="stretch", type="primary")
    c[1].button("exclude", on_click=decide, args=(paths, order, nid, "exclude", pano_id), width="stretch")
    c[2].button("keep", on_click=decide, args=(paths, order, nid, "keep", pano_id), width="stretch",
                help="use this node even though the pipeline flagged it")
    c[3].button("undo", on_click=undo, args=(paths,), width="stretch", disabled="undo" not in st.session_state,
                help="put back the last decision")
    c[5].button("◀ prev", on_click=step, args=(order, -1), width="stretch")
    c[6].button("next ▶", on_click=step, args=(order, 1), width="stretch")

    frames = manifest[manifest.node_id == nid].sort_values("heading")
    for col, f in zip(st.columns(max(len(frames), 1)), frames.itertuples()):
        col.image(str(paths.raw_photos / f.file), caption=f"{f.heading:g}°", width="stretch")
    if renders is not None:
        views = renders[renders.node_id == nid]
        for col, v in zip(st.columns(max(len(views), 1)), views.itertuples()):
            col.image(str(paths.views / v.file), caption=f"180° view, {v.walk.replace('_', ' ')}",
                      width="stretch")


if __name__ == "__main__":                                          # streamlit runs it
    cfg = load_config(CONFIG)
    main(cfg, Paths(REPO, cfg))
