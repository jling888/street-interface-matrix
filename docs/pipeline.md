# Pipeline Guide

`src/main.py` runs the study's steps in order, in five stages, for one experiment.

## Setup

Install the dependencies:
```
uv sync
```

Store your Google Maps key for the Street View frames (`.env` is git-ignored):
```
echo GMAPS_KEY=<your_key> > .env
```

## Steps

Run every step of the default experiment (`murrayhill_qwen3`):
```
uv run python src/main.py
```

Run one stage or one step:
```
uv run python src/main.py <stage_or_step>
```

Run another experiment, all of it or one stage or step:
```
# Run all
uv run python src/main.py configs/experiments/<experiment_config_name>.yaml

# Run one stage or one step
uv run python src/main.py configs/experiments/<experiment_config_name>.yaml <stage_or_step>
```

Review the nodes in the browser:
```
uv run streamlit run src/data_prep/node_review_app.py
```

## Stages and steps

A failing step stops the run. The GPU steps can run on Colab instead ([`colab.md`](./colab.md)).

| Stage | Step | Description | Required |
|---|---|---|---|
| `data_prep` | `sample_street_nodes` | Nodes at a fixed spacing along the study's OSM streets, with heading and walk order; a map of them | OSM (downloaded) |
| | `download_street_view` | Street View metadata and the frames per node | `GMAPS_KEY` in `.env`, for new frames |
| | `stitch_180_views` | The views per node, one per walking direction, stitched from the frames | `download_street_view` |
| `inference` | `run_segmentation_models` | The segmentation models on every photo; their raw outputs | GPU |
| | `rate_views_with_vlm` | The VLM's rating of every term per view, and the expected rung | `stitch_180_views`; GPU |
| | `fuse_label_maps` | One class per pixel from the raw outputs | `run_segmentation_models` |
| | `calculate_segmentation_pixel_shares` | Class shares per node and bearing (whole view, eye-level band); GVI, VEI; block faces | `fuse_label_maps` |
| `public_data` | `measure_street_canyons` | Building height H, facade-to-facade width W, H/W, the canyon's sky view factor; the buildings with height | `calculate_segmentation_pixel_shares`; the city's building footprints (downloaded) |
| | `measure_city_records` | Land use, seating and the street's features per node, from the city's records | `calculate_segmentation_pixel_shares`, `measure_street_canyons`; the city's datasets (downloaded) |
| | `measure_space_syntax` | Angular choice and integration at the configured radii; walking distances between nodes | `sample_street_nodes`; the OSM walking network (downloaded) |
| | `measure_pedestrian_volume` | The outcome: modelled pedestrian volume per node, checked against observed counts | The city's pedestrian volumes (downloaded) |
| `metric` | `compose_metric` | The metric's score M per view, from its indicators; its weights, robustness, coherence and factor loadings | `rate_views_with_vlm`, `measure_city_records` |
| `analysis` | `validate_vlm_ratings` | The ratings against the segmentation over each view's arc, the canyons and the records; repeatability; a suggested role per agreement term | `rate_views_with_vlm`, `calculate_segmentation_pixel_shares`, `measure_street_canyons`, `measure_city_records` |
| | `fit_footfall_regression` | M against footfall: OLS (network only, Models 1-3, Model 3 by indicator) with plain, clustered, Conley and wild-bootstrap inference; SEM, lag, SLX, ESF | `compose_metric`, the `public_data` stage |
| | `cluster_street_types` | k-means street types; stability, silhouette, MANOVA, Tukey HSD on measures the clustering does not use | `compose_metric`, the `public_data` stage |
| | `map_dwell_potential` | Predicted lingering, footfall × (M / 7)^s at the configured exponents; mapped by street | `compose_metric`, `measure_pedestrian_volume` |
| | `make_figures` | The figures at printed size (`map_style.PAPER_RC`) | Whatever exists |
| | `export_all_results` | Every result per node in one GeoJSON (for QGIS) and CSV | The steps before |
| | `make_results_viewer` | One page to explore every per-node result in a browser, 2D or 3D | `export_all_results` |

## Configs

An experiment, `configs/experiments/<run_id>.yaml`, picks one config per stage.

| Stage | Config | Description |
|---|---|---|
| `experiments` | `configs/experiments/murrayhill_qwen3.yaml` | The study: name, city, CRS, seed, `osm_date`, and one config per stage |
| | `configs/experiments/murrayhill_jev.yaml` | Inherits the study, with `inference: jev` |
| `data_prep` | `configs/data_prep/murrayhill.yaml` | Study area, street list, typology; nodes every 20 m; four 90-degree frames per node (headings square to the grid); two 180-degree views per node |
| `inference` | `configs/inference/qwen3.yaml` | Segmentation (variant `mask2former-sam3`: Mask2Former on Mapillary Vistas and ADE20K, and SAM 3; fusion), the VLM (`model_file`, Qwen3-VL-32B, which needs an 80 GB GPU), the pixel shares |
| | `configs/inference/jev.yaml` | Inherits `qwen3.yaml`, with JEV-27B-VL |
| | `configs/inference/segmentation_classes.yaml` | The classes, each model's labels for them, the SAM 3 prompts, the terms |
| | `configs/inference/vlm_prompts.yaml` | The VLM's 31 terms and prompts, each rated 1-7 |
| `public_data` | `configs/public_data/new_york.yaml` | Canyon, city-record, Space Syntax (radii 400, 800 and 1,600 m) and footfall measures (radii, rays, ...) |
| `metric` | `configs/metric/street_interface_matrix/imageability.yaml` | M in Human Development Index form: the indicators between goalposts, three dimensions, their geometric mean |
| `analysis` | `configs/analysis/footfall_and_types.yaml` | Validation thresholds, the regression's controls and inference, the clusters (k = 3, Tukey on volume, H/W and integration), the dwell map (s = 0.5, 1, 2; mapped at s = 1) |

A config or experiment may `inherit` another and change part of it. The run id is the experiment file's name: results
go to `outputs/<run_id>/`, with the settings used in `config.yaml`.

Experiments share the ratings of their inference config and the segmentation of their `segmentation.variant`, so
changing only the metric or analysis needs no GPU step. After changing the VLM's prompts or model, the saved ratings
are refused: delete `view-ratings.csv` to rate again.

## Variants

A variant is a sibling chosen by name in the config, writing the same files, so the later stages work unchanged.

| Variant | Add | Chosen by |
|---|---|---|
| VLM | A file in `inference/vlm/` with one `VisionLanguageModel` subclass (`build_prompt`, `load_model`) | `vlm.model_file` |
| Segmentation checkpoint | Another checkpoint for a model's role in `fuse` (`mapillary`, `ade20k`, `sam3`), with the same labels, under a new `segmentation.variant` | `segmentation.models` |
| Segmentation model | A `SegmentationModel` subclass (`is_complete`, `load_model`, `run_on_photo`) with a reader for its outputs; add it in `run_segmentation.py`, give it its place in `fuse` and its labels in `segmentation_classes.yaml` | The code |
| City | A folder in `public_data/` with the four `public_data` steps, writing the same files as `public_data/new_york/` (its per-node files, `buildings-with-height.gpkg` and `map-layers.json`) | The experiment's `public_data` |
| Metric | A folder in `metric/` with a `run(cfg, paths)` writing `scores-per-usable-view.csv`; its config's `metric` section gives the `dimensions` and `names` the analysis reads | The experiment's `metric_system` |
