"""!
@package run_vlm
@brief VLM ratings: the terms of configs/inference/vlm_prompts.yaml per view on a seven-rung scale, read from
       the rung probabilities.

The model is the file in vlm/ that vlm.model_file names (qwen3: Qwen3-VL-32B; jev_27b: JEV-27B-VL, for a
later paper), a VisionLanguageModel (vlm/base.py): every prompt says the view, the term's definition,
what to look at and its seven named rungs (the city is never named); the prompts of one image go in one batch, each
answer forced to begin with the rung, whose probability over 1-7 is read from the logits. The rating is the
probability-weighted mean rung (the scoring rule of G-Eval, an LLM-as-a-judge method). Here, the same for every model:
  1. the model   the one the config names
  2. one view    the images its terms are rated on, a facade's two sides combined, its row of the table
  3. rate_views  rates every view of the usable nodes in views-180.csv (resumable)
The ratings' agreement with independent data is step validate_vlm_ratings (analysis/vlm_rating_validation.py).

Terms with view: along are rated on the 180-degree view; facade terms on the two level frames square to the frontage
(left, right), stored per side and combined by mean or max (combine_sides). A term with cannot_judge has a rung 0, read
separately, so its rating stays the expected rung over 1-7.

Stored per term: the expected value (_ev, the score the Street Interface Matrix uses), round(EV), _argmax and _p1.._p7,
with each side's rating and rung 0 for the facade terms. Rating needs a CUDA GPU. Writes
data/<city>/inference/vlm/<model setup>/:
view-ratings.csv and view-ratings-rated-with.json (the model and prompts that made it). Run by
pipeline step rate_views_with_vlm.
"""
import importlib
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from common.data_readers import rated_fields
from common.node_table import read_nodes
from inference.vlm.base import RUNGS, VisionLanguageModel, is_facade_term, term_definitions

ID = ["file", "street", "walk", "seq", "node_id", "cardinal"]   # a stored row's identity columns


# ---------------------------------------------------------------- 1. the model
def vlm_model(V):
    """!
    @brief The VisionLanguageModel of the file in vlm/ that vlm.model_file names (not loaded yet).
    """
    module = importlib.import_module(f"inference.vlm.{V['model_file']}")
    (model,) = [c for c in vars(module).values()
                if isinstance(c, type) and issubclass(c, VisionLanguageModel) and c is not VisionLanguageModel]
    return model(V)


# ---------------------------------------------------------------- 2. one view
def combine_sides(P, V, cannot=None):
    """!
    @brief One distribution per field from rung_probabilities(): a facade term's two sides combined by its `sides` rule,
           mean (the mixture of the two distributions: its expected value is the mean of the two) or max (the side
           with the higher expected value: one good side is enough to stop). A side the model cannot judge
           (P(rung 0) above vlm.cannot_judge_above) is left out; with neither side, the distribution is NaN.
    @param cannot  The model's rung_probabilities(), second result, or None.
    """
    out = {k: p for k, p in P.items() if not isinstance(k, tuple)}
    for f in dict.fromkeys(k[0] for k in P if isinstance(k, tuple)):
        seen = [P[(f, s)] for s in "LR" if (cannot or {}).get((f, s), 0) <= V["cannot_judge_above"]]
        rule = term_definitions(V)[f]["sides"]
        if not seen:
            out[f] = np.full(7, np.nan)
        elif rule == "mean":
            out[f] = sum(seen) / len(seen)
        elif rule == "max":
            out[f] = max(seen, key=lambda p: p @ RUNGS)
        else:
            raise ValueError(f"{f}: sides must be mean or max, not {rule}")
    return out


SIDE_NAMES = {"L": "left", "R": "right"}


def record(r, P, sides=None, cannot=None):
    """!
    @brief One row of view-ratings.csv.
    @param r       The view's row of views-180.csv.
    @param P       combine_sides() output.
    @param sides   The model's rung_probabilities(), first result, for each facade term's two sides, or None.
    @param cannot  The model's rung_probabilities(), second result, or None.
    @return {column: value}: identity, then per field round(EV), _ev (the rating: the probability-weighted
            mean rung, NaN if neither side could be judged), _argmax, _p1.._p7; per side <field>_left_rating and,
            with cannot_judge, <field>_left_p0 (likewise right).
    """
    rec = {c: getattr(r, c) for c in ID}
    for f, p in P.items():
        ev = float(p @ RUNGS)
        judged = not np.isnan(ev)
        rec[f] = int(np.clip(round(ev), 1, 7)) if judged else np.nan
        rec[f + "_ev"] = ev
        rec[f + "_argmax"] = int(RUNGS[p.argmax()]) if judged else np.nan
        rec.update({f"{f}_p{k}": round(float(q), 4) for k, q in zip(RUNGS, p)})
    for k, p in (sides or {}).items():
        if isinstance(k, tuple):
            rec[f"{k[0]}_{SIDE_NAMES[k[1]]}_rating"] = float(p @ RUNGS)
    rec.update({f"{k[0]}_{SIDE_NAMES[k[1]]}_p0": round(q, 4) for k, q in (cannot or {}).items()})
    return rec


def nearest_heading(headings, bearing):
    """!
    @brief The frame heading closest to `bearing` (degrees, either way round).
    """
    return min(headings, key=lambda h: abs((h - bearing + 180) % 360 - 180))


def view_images(r, paths, V, headings):
    """!
    @brief The images a view's terms are rated on: the stitched view; for facade terms (is_facade_term) the two level
           frames nearest the bearing -90 (left) and +90 (right), keyed (field, side).
    """
    im = Image.open(paths.views / r.file).convert("RGB")
    frame = lambda b: Image.open(paths.raw_photos / f"{r.node_id}_{nearest_heading(headings, b):03d}.jpg").convert("RGB")
    sides = {"L": frame(r.bearing - 90), "R": frame(r.bearing + 90)}
    out = {}
    for f in term_definitions(V):
        if is_facade_term(f, V):
            out.update({(f, s): i for s, i in sides.items()})
        else:
            out[f] = im
    return out


def rate_one(model, r, paths, V, headings):
    """!
    @brief One view's row of view-ratings.csv.
    @param model  A loaded VisionLanguageModel.
    """
    P, cannot = model.rung_probabilities(view_images(r, paths, V, headings))
    return record(r, combine_sides(P, V, cannot), P, cannot)


def save(done, out, path):
    """!
    @brief Write the ratings so far, sorted, in a fixed column order.
    @return The table written.
    """
    d = pd.concat([t for t in (done, pd.DataFrame(out)) if len(t)], ignore_index=True)
    d = d.sort_values(["street", "walk", "seq"])
    F = rated_fields(d)
    cols = ID + F + [f + "_ev" for f in F] + [f + "_argmax" for f in F] + [f"{f}_p{k}" for f in F for k in RUNGS]
    d = d.reindex(columns=[*cols, *(c for c in d.columns if c not in cols)])
    d.to_csv(path, index=False)
    return d


# ---------------------------------------------------------------- 3. rate the views
def rate_views(cfg, paths):
    """!
    @brief Rate the usable nodes' views not yet in view-ratings.csv (resumable; saved every 25 views).
    """
    V = cfg["vlm"]
    table = paths.ratings_csv
    headings = cfg["download_gsv_panos"]["headings"]

    fl = pd.read_csv(paths.views_csv)
    # only usable nodes (read_nodes: review, the imagery check)
    usable = read_nodes(paths).query("usable").node_id
    print(f"rating the views of {usable.nunique()} usable nodes; {(~fl.node_id.isin(usable)).sum()} views skipped")
    fl = fl[fl.node_id.isin(usable)]
    # a table holds one model's ratings with one prompt set: resuming with others would mix instruments
    stamp = table.with_name(table.stem + "-rated-with.json")
    made = json.loads(stamp.read_text()) if stamp.exists() else None
    model = vlm_model(V)
    want = {"model": V["model"], "prompts": model.prompts_fingerprint()}
    if table.exists() and made and any(made.get(k) != v for k, v in want.items()):
        sys.exit(f"{table} was rated with {made}, not {want}: delete it to rate again with the current prompts")
    done = pd.read_csv(table) if table.exists() else pd.DataFrame()
    fl = fl[~fl.file.isin(set(done.file))] if len(done) else fl
    print(f"{len(fl)} views to rate, {len(done)} already done")
    if len(fl):
        model.load_model()
        stamp.write_text(json.dumps(want))
        out = []
        for r in tqdm(list(fl.itertuples()), desc="views", mininterval=10.0):
            out.append(rate_one(model, r, paths, V, headings))
            if len(out) % 25 == 0:
                save(done, out, table)
        d = save(done, out, table)
        print(f"{len(d)} rows -> {table}")
        for f in term_definitions(V):
            mode = (d[f] == d[f].mode().iloc[0]).mean() * 100
            print(f"  {f:<23} mean {d[f].mean():.2f}  sd {d[f].std():.2f}  modal {mode:.0f}%")

    # colab.sh watch stops at this line
    rated = len(pd.read_csv(table)) if table.exists() else 0
    print(f"rate_views_with_vlm done: {rated} views in {table.name}")
