# Apps Guide

Three tools in `src/apps/`, beside the pipeline: they read its files and use its code, and the pipeline never runs
them.

| App | What it is for | Runs |
|---|---|---|
| `node_review_app.py` | Look at each node's Street View imagery and record whether it is usable: ok / exclude / keep | Streamlit, locally |
| `validation_app.py` | Whether each VLM rating follows what the photo shows, term by term and view by view | Streamlit, locally |
| `fast_inference.py` | Segment any folder of images with the study's models, outside the pipeline | Python script, with `fast_inference.sh` |

## Node review

Review each node's imagery before the GPU steps, so excluded nodes are not rated:
```
uv run streamlit run src/apps/node_review_app.py
```

It shows one node at a time: the panorama's date, photographer and distance, the pipeline's verdict, the decision
buttons, the four Street View frames, and the two 180-degree views. Their segmentation is checked in the validation
app.

| Control | What it does |
|---|---|
| `ok` / `exclude` / `keep` | Saves the decision at once to `data/<city>/data_prep/node-review-decisions.csv`, which every step applies, and moves to the next node. `keep` uses a node the pipeline flagged |
| Reason | A common reason to pick, and free text after it; both are saved with `exclude` and `keep` |
| `undo` | Puts back what the last decision replaced, and returns to its node |
| `◀ prev` / `next ▶` | Moves within the filter without deciding |

Each decision records the panorama it was made on (`pano_id`). If the node's panorama is replaced later (a new
download, or a re-probe), the decision counts as not reviewed: the node shows under `not reviewed yet`, and the sidebar
counts it as imagery changed since review. Decisions saved before this was recorded are not checked.

| Filter | Shows |
|---|---|
| `not reviewed yet` | Nodes without a decision, or with one made on a panorama since replaced |
| `flagged by the pipeline` | Nodes the download step found no usable imagery for: no panorama, too far from the node, a user upload with no Google panorama near, a failed download. `keep` uses one anyway |
| `re-probed (panorama off the node)` | Nodes whose panorama was taken a few metres along the street, as the nearest was a user upload |
| `excluded by review` | Nodes excluded by hand |

## Validation

Check the ratings of an experiment:
```
uv run streamlit run src/apps/validation_app.py
```

It answers one question: does each VLM rating follow what the photo shows? Choose the experiment and whether to count
only usable nodes in the sidebar, and `Walk` or `Terms` at the top.

| View | Shows |
|---|---|
| `Walk` | One street, view by view, as a pedestrian walks it: the 180-degree view, a strip of M along the walk, every rating grouped by the experiment's pillars beside what was measured for that view (in a box that scrolls, so M below stays in view), M and its pillars, and where the view sits among all views. Step with ◀ ▶ or the slider, `play` to walk on its own, or turn round with `Walking` |
| `Terms` | Every rated term in plain words: what it is checked against, how far the two agree over all views, whether the rating repeats on the next view along the walk, how bunched it is, and a verdict. Hover over a column's header for what it means |

On `Walk`, each rating has a line under it: what was measured for the same view (the segmented share over the view's
arc, or the node's record) and whether the two agree, by how far apart they sit among all views (percentile ranks):
`✓ agrees` under 0.25 apart, `⚠ disagrees` 0.5 or more, `~ apart` between. Terms nothing measures have no line;
`Not in the score` lists the rated terms the experiment's metric does not use. The node's four frames with their
segmentation are under the view, folded. M and its pillars come from the experiment's `compose_metric` step
(`outputs/<experiment>/metric/`), so they differ between `murrayhill_qwen3_v1` and `murrayhill_qwen3_v2`; the ratings
are the same for both.

On `Terms`, tick a term at the left of its row to open it below the table: its definition and seven rungs, its rating
against what it is checked against (a scatter, and the counterpart per rung), and the views where the two disagree
most. Tick one of those views to walk to it: `Walk` opens on its street and direction, at that view.

| Verdict | Agreement (Spearman's rho, turned so higher is better) |
|---|---|
| `✓ follows the photo` | 0.5 and above |
| `~ weak` | 0.3 to 0.5 |
| `✗ does not follow` | Under 0.3 |
| `(bunched)` | Over 70% of the views on one rung (`max_modal_share`) |
| `(not repeatable)` | Correlates under 0.5 with the next view along the walk (`min_repeatability`) |

The agreement thresholds are a reading, not a test; the other two are `validate_vlm_ratings`' own, in
`configs/analysis/footfall_and_types.yaml`. The ratings, their marks on `Walk` and every figure on `Terms` follow the
sidebar's choice of views; M's strip and where a view sits are among all the views the metric scored.

| Data | Read from |
|---|---|
| Ratings, rung probabilities | `data/<city>/inference/vlm/<model>/view-ratings.csv` (`rate_views_with_vlm`) |
| M and its pillars (`Walk`) | `outputs/<experiment>/metric/street_interface_matrix/scores-per-usable-view.csv` (`compose_metric`) |
| Segmented share of each view | `data/<city>/inference/segmentation/<variant>/measurements/term-shares-by-bearing.npz` (`calculate_segmentation_pixel_shares`) and `data/<city>/data_prep/views-180.csv` |
| Records, usable flag | Each node's measurements (`public_data` stage, `node-review-decisions.csv`), as the pipeline joins them |
| Images, label maps | `data/<city>/data_prep/views-180/`, the Street View frames, and `.../segmentation/<variant>/label-maps/` |
| Term names, definitions, rungs | `configs/inference/vlm_prompts.yaml` |

It computes every figure from these, so it needs the pipeline up to `calculate_segmentation_pixel_shares`, the
`public_data` stage for the records, and `compose_metric` for M on `Walk`; not `validate_vlm_ratings`.

To look at a `fast_inference` run instead, give its output folder:
```
uv run streamlit run src/apps/validation_app.py -- --output_path <image folder>-segmented
```
The page then shows each image's class shares, and its label map over it.

## Fast inference

Segment any folder of images with the study's segmentation: the three models and their fusion, with the settings of
`configs/inference/`. Its label maps match the study's for the same image.

Install and sign in once, as for the pipeline ([`pipeline.md`](./pipeline.md), [`colab.md`](./colab.md)): SAM 3 needs a
Hugging Face account with access to it.
```
uv sync
.venv/bin/hf auth login
```

Try it on three images, then run the folder:
```
src/apps/fast_inference.sh <image folder> --limit 3
src/apps/fast_inference.sh <image folder>
```

A run cut short resumes where it stopped: run the same command again. An image that cannot be read or run is logged in
`errors.csv`, and the run goes on.

### Arguments

| Argument | Description | Default |
|---|---|---|
| `<image folder>` | The images, at any depth: `.jpg`, `.jpeg`, `.png`, `.webp`, `.tif`, `.tiff`, `.bmp` | **Required** |
| `--out` | The output folder | `<image folder>-segmented` |
| `--exp` | The experiment whose segmentation settings to use: `configs/experiments/<name>.yaml` | `murrayhill_qwen3_v2` |
| `--limit` | Only the first N images, in path order | All |

### Output

| File | Content |
|---|---|
| `label-maps/<image>.png` | One class id per pixel; 255 is ignored (the capture vehicle) |
| `label-taxonomy.csv` | Class id → class name |
| `overlays/<image>.png` | The label map over the image, in the validation app's colours |
| `class-shares.csv` | One row per image: each class's share of the pixels not ignored, in % |
| `errors.csv` | The images that failed, with the error |
| `label-settings.json` | The segmentation settings used: a later run into the folder must use the same, or stops |
| `raw-model-outputs/` | Each model's own output per image, from which the label maps are fused and runs resume |

An image is named by its path under the folder, with `/` as `__`: `sub/a.jpg` → `sub__a.png`.

### Speed

| Device | Use |
|---|---|
| CUDA GPU | Folders of any size |
| Apple GPU (MPS) | A few images: a test, or a look |
| CPU | A few images, slowly |

The script picks the first available, in that order. It loads the models once, keeps each model's output as soon as it
is made, and skips what is done. It uses the study's models at full precision; lighter models or half precision would
be faster but would no longer match the study's label maps.
