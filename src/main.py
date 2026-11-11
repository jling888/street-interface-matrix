"""The experiment, stage by stage: data_prep -> inference -> public_data -> metric -> analysis.

    python src/main.py                                    # every step of the study
    (configs/experiments/murrayhill_qwen3_v2.yaml)
    python src/main.py inference                          # one stage
    python src/main.py compose_metric                     # one step, by name
    python src/main.py configs/experiments/murrayhill_jev.yaml    # another experiment

Every step runs in its own process, in the order of STAGES; a failing step stops the run. Steps download what they
need the first time (download_street_view needs GMAPS_KEY in .env), and log each file in
data/<city>/downloaded-sources.csv; run_segmentation_models and rate_views_with_vlm need a GPU for photos or views
without results (src/colab.sh).

The five stages, one folder each:
  data_prep/          where to look and what is there: the sampling nodes and the Street View views (any city)
  inference/          every model run on the photos: segmentation and the VLM (any city); a model's own code is a
                      sibling in segmentation/ or vlm/, chosen by its config
  public_data/        what a city's records and geometry say per node: one folder per city (new_york/), its sources
                      as constants
  metric/             a score from the ratings and the records: one folder per system (street_interface_matrix/)
  analysis/           the ratings checked, and what the score explains (any city, any metric system)
  common/             settings and paths, downloads, the readers of the files one stage writes for the next, styling
An experiment picks one config per stage (and, for the public data and the metric, its code). Every step is one
function of its settings (`cfg`) and its files (`paths`); the stages share files, never code: a stage imports only
from common/. src/ is the import root: running this file puts it on the path.
"""
import importlib
import subprocess
import sys

from common.settings_and_paths import CONFIG, start

# stage: [(step, module, its function of (cfg, paths))]; {public_data} and {metric_system} are the experiment's
STAGES = {
    "data_prep": [
        ("sample_street_nodes", "sample_osm_nodes", "run"),
        ("download_street_view", "download_gsv_panos", "run"),
        ("stitch_180_views", "stitch_180_views", "run"),
    ],
    "inference": [
        ("run_segmentation_models", "run_segmentation", "run_models"),
        ("rate_views_with_vlm", "run_vlm", "rate_views"),
        ("fuse_label_maps", "fuse_segmentation_labels", "fuse_label_maps"),
        ("calculate_segmentation_pixel_shares", "calculate_segmentation_pixel_shares", "run"),
    ],
    "public_data": [
        ("measure_street_canyons", "{public_data}.measure_street_canyons", "run"),
        ("measure_city_records", "{public_data}.measure_city_records", "run"),
        ("measure_space_syntax", "{public_data}.measure_space_syntax", "run"),
        ("measure_pedestrian_volume", "{public_data}.measure_pedestrian_volume", "run"),
    ],
    "metric": [
        ("compose_metric", "{metric_system}.{metric_system}", "run"),
    ],
    "analysis": [
        ("validate_vlm_ratings", "vlm_rating_validation", "run"),
        ("fit_footfall_regression", "footfall_regression", "run"),
        ("cluster_street_types", "street_type_clusters", "run"),
        ("map_dwell_potential", "dwell_potential_map", "run"),
        ("make_figures", "paper_figures", "run"),
        ("export_all_results", "results_table_export", "run"),
        ("make_results_viewer", "results_viewer", "build"),
    ],
}
# every step by name, in run order: the names the steps' logs and colab.sh use
STEPS = {step: (stage, module, function) for stage, steps in STAGES.items() for step, module, function in steps}


def find_steps(wanted):
    """!
    @brief The steps a name stands for: a stage (`inference`) or one step (`rate_views_with_vlm`).
    """
    if wanted in STAGES:
        return [name for name, (stage, _, _) in STEPS.items() if stage == wanted]
    if wanted in STEPS:
        return [wanted]
    sys.exit(f"no step or stage {wanted}: one of {', '.join(STAGES)}, {', '.join(STEPS)}")


def run_step(name, config):
    stage, module, function = STEPS[name]
    cfg, paths = start(name, config)
    module = module.format(public_data=cfg["public_data"], metric_system=cfg["metric_system"].split("/")[0])
    getattr(importlib.import_module(f"{stage}.{module}"), function)(cfg, paths)


if __name__ == "__main__":
    configs = [a for a in sys.argv[1:] if a.endswith((".yaml", ".yml"))]
    wanted = [a for a in sys.argv[1:] if a not in configs]
    names = [n for w in wanted for n in find_steps(w)] if wanted else list(STEPS)
    if len(names) == 1:
        run_step(names[0], configs[0] if configs else CONFIG)
    else:
        for name in names:
            if subprocess.run([sys.executable, __file__, name, *configs]).returncode:
                sys.exit(f"{name} failed")
