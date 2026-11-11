"""!
@package settings_and_paths
@brief The settings of an experiment, and where every file lives.

An experiment (configs/experiments/<run_id>.yaml) names the run, its city and one config per stage, and replaces any
settings it lists:
    data_prep: murrayhill                 configs/data_prep/murrayhill.yaml (study area, streets, headings)
    inference: qwen3                      configs/inference/qwen3.yaml (models, thresholds)
    public_data: new_york                 configs/public_data/new_york.yaml, and the code in public_data/new_york/
    metric_system: street_interface_matrix/identity_dependence_comfort
                                          configs/metric/street_interface_matrix/identity_dependence_comfort.yaml, and
                                          the code in metric/street_interface_matrix/
    analysis: footfall_and_types          configs/analysis/footfall_and_types.yaml
A config's sections are named after the files that read them; a config or an experiment may `inherit` another and
change some of it. With them come configs/inference/segmentation_classes.yaml and vlm_prompts.yaml, checked on load.

Paths names every file, from the repo root and the config; library code only ever gets a Paths, never a path of its
own:
    data/<city>/<stage>/      what depends only on the city (and the model setup): shared by every experiment; a
                              stage's downloads/ is what it fetched and never changes
    outputs/<run_id>/         one VLM and scoring run: its config, ratings, scores and analyses
"""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
CONFIG = CONFIGS / "experiments" / "murrayhill_qwen3_v2.yaml"       # the study
# experiment key -> its configs/ folder
STAGE_CONFIGS = {"data_prep": "data_prep", "inference": "inference", "public_data": "public_data",
                 "metric_system": "metric", "analysis": "analysis"}


# ---------------------------------------------------------------- the settings
PROMPT_KEYS = ("system", "view_sentences", "instructions", "choose_rung", "choose_option", "scale_and_reply", "terms")
PROMPTS_READ = {"enclosure"}                        # terms the code reads directly (validate_vlm_ratings)


def load_prompts(path):
    """!
    @brief configs/inference/vlm_prompts.yaml, checked: the sections, the terms the code reads, and each term's fields.
    @return The file as a dict; raises ValueError naming every problem.
    """
    t = yaml.safe_load(Path(path).read_text())
    problems = [f"missing section {k}" for k in PROMPT_KEYS if k not in t]
    terms = t.get("terms", {})
    problems += [f"the code reads term {k}" for k in PROMPTS_READ - set(terms)]
    for name, c in terms.items():
        problems += [f"{name}: missing {k}" for k in ("view", "definition", "check", "scale") if k not in c]
        if len(c.get("scale", [])) != 7:
            problems.append(f"{name}: scale must have seven rungs")
        if c.get("view") not in ("along", "facade"):
            problems.append(f"{name}: view must be along or facade")
        if c.get("view") == "facade" and c.get("sides") not in ("mean", "max"):
            problems.append(f"{name}: a facade term's sides must be mean or max")
        if "cannot_judge" in c and c.get("view") != "facade":
            problems.append(f"{name}: cannot_judge is for facade terms only")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return t


def merged(base, own):
    """!
    @brief `own` over `base`: a section's keys replace the same keys of that section; anything else replaces whole.
    """
    return {**base, **{k: {**base.get(k, {}), **v} if isinstance(v, dict) else v for k, v in own.items()}}


def load_yaml(path):
    """!
    @brief A config or experiment file, merged over the one its `inherit` names (beside it).
    """
    own = yaml.safe_load(Path(path).read_text())
    return merged(load_yaml(Path(path).parent / own.pop("inherit")), own) if "inherit" in own else own


def load_config(path):
    """!
    @brief The settings of an experiment: the config of each stage it names, then its own settings, with the classes
           (cfg["classes"], ["ignore"], ["terms"]) and the prompts (cfg["vlm"]["prompts"]).
    @param path  The experiment file; its name, without .yaml, is `run_id`, the outputs/<run_id>/ folder.
    @return dict.
    """
    experiment, cfg = load_yaml(path), {}
    for key, folder in STAGE_CONFIGS.items():
        cfg = merged(cfg, load_yaml(CONFIGS / folder / f"{experiment[key]}.yaml"))
    cfg = merged(cfg, experiment)
    cfg["run_id"] = Path(path).stem
    model = CONFIGS / "inference"
    cfg["vlm"] = {**cfg["vlm"], "prompts": load_prompts(model / "vlm_prompts.yaml")}
    return {**cfg, **load_classes(model / "segmentation_classes.yaml")}


# ---------------------------------------------------------------- the classes
KEYS = ("classes", "ignore", "terms")               # the file's sections; load_config() adds them to every config

# names the code reads directly; renaming one in segmentation_classes.yaml must fail here, not mid-run
CLASSES_READ = {"sky"}                             # the visible sky share (validate_vlm_ratings)
TERMS_READ = {"vegetation", "sky", "building",      # GVI and VEI (TermSharesByBearing.gvi_and_vei), arc shares
              "V_pave",                             # label-map metrics (PVI_band)
              "V_nat", "V_built", "signage", "openings", "seating"}   # vlm_rating_validation.ANCHOR


def load_classes(path):
    """!
    @brief configs/inference/segmentation_classes.yaml: `classes` in priority order (first listed wins), `ignore`,
           and `terms` (each a union of classes).
    @return {"classes": ..., "ignore": ..., "terms": ...}; raises ValueError naming every class or term that is
            missing.
    """
    t = yaml.safe_load(Path(path).read_text())
    classes, terms = set(t["classes"]), set(t["terms"])
    problems = []
    named = {r for spec in t["classes"].values() for r in spec.get("may_replace", [])} - classes - {"unknown"}
    if named:
        problems.append(f"may_replace names classes not in classes: {sorted(named)}")
    in_terms = {c for cs in t["terms"].values() for c in cs} - classes
    if in_terms:
        problems.append(f"terms name classes not in classes: {sorted(in_terms)}")
    if CLASSES_READ - classes:
        problems.append(f"classes the code reads are missing: {sorted(CLASSES_READ - classes)}")
    if TERMS_READ - terms:
        problems.append(f"terms the code reads are missing: {sorted(TERMS_READ - terms)}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return {k: t[k] for k in KEYS}


# ---------------------------------------------------------------- where every file lives
class Paths:
    """!
    @brief Every folder and file, as attributes. By stage: what depends only on the city (and the model setup) is in
           data/<city>/<stage>/, shared by every experiment; what depends on the experiment's metric and analysis is in
           outputs/<run_id>/<stage>/. A stage's downloads/ holds what it fetched and never changes: the files to keep,
           and not to share.
    """

    def __init__(self, root, cfg):
        """!
        @param root  The repository root.
        @param cfg   The experiment's settings: its `city`, `run_id`, `inference` (the model setup, whose ratings are
                     stored under its name), `metric_system` and segmentation.variant.
        """
        root = Path(root)
        city = root / "data" / cfg["city"]
        run = root / "outputs" / cfg["run_id"]
        self.root, self.city_data = root, city
        self.downloads_log = city / "downloaded-sources.csv"            # every file downloaded, when, its sha256

        # data/<city>/data_prep/: by hand, downloaded, and the nodes and views made from them
        prep = self.data_prep = city / "data_prep"
        self.node_review = prep / "node-review-decisions.csv"           # ok / exclude / keep, by hand (the review app)
        # the study area is the config's sample_osm_nodes.polygon, in this folder (by hand)
        self.data_prep_downloads = prep / "downloads"
        self.data_prep_osm_cache = self.data_prep_downloads / "openstreetmap-cache"   # Overpass responses by query hash
        self.raw_photos = self.data_prep_downloads / "street-view-frames"               # <node_id>_<heading>.jpg
        self.raw_photos_csv = self.data_prep_downloads / "street-view-frames.csv"       # one row per downloaded frame
        self.pano_metadata_csv = self.data_prep_downloads / "street-view-metadata.csv"  # one row per node: its panorama
        self.nodes_gpkg = prep / "nodes.gpkg"
        self.nodes_csv = prep / "nodes.csv"
        self.streets_gpkg = prep / "study-streets.gpkg"
        self.nodes_map = prep / "nodes-map.png"
        self.views = prep / "views-180"                                 # <street>/<walk>/<seq>_<node>_<N|E|S|W>.jpg
        self.views_csv = prep / "views-180.csv"                         # one row per view, `file` under views/

        # data/<city>/inference/: per segmentation variant, and the ratings per model setup
        seg = city / "inference" / "segmentation" / cfg["segmentation"]["variant"]
        self.raw_model_outputs = seg / "raw-model-outputs"
        self.fused_label_maps = seg / "label-maps"
        self.fused_label_settings = seg / "label-settings.json"
        self.fused_label_taxonomy = seg / "label-taxonomy.csv"
        self.label_map_measurements = seg / "measurements"
        self.term_shares_by_bearing_npz = self.label_map_measurements / "term-shares-by-bearing.npz"
        self.class_shares_csv = self.label_map_measurements / "class-shares-per-node.csv"
        self.view_indices_csv = self.label_map_measurements / "view-indices-per-node.csv"
        self.block_faces_csv = self.label_map_measurements / "view-indices-per-block-face.csv"
        self.vlm_ratings = city / "inference" / "vlm" / cfg["inference"]
        self.ratings_csv = self.vlm_ratings / "view-ratings.csv"        # and view-ratings-rated-with.json beside it

        # data/<city>/public_data/: the city's downloads, and what they say per node
        pub = city / "public_data"
        self.public_data_downloads = pub / "downloads"
        self.public_data_osm_cache = self.public_data_downloads / "openstreetmap-cache"
        self.walking_network = self.public_data_downloads / "walking-network.osm"     # the network Space Syntax reads
        self.street_canyons_csv = pub / "street-canyons-per-node.csv"
        self.buildings_with_height = pub / "buildings-with-height.gpkg"  # for any later stage, any city
        self.land_use_csv = pub / "land-use-per-node.csv"
        self.city_records_csv = pub / "city-records-per-node.csv"
        self.space_syntax_csv = pub / "space-syntax-per-node.csv"
        self.walking_distances = pub / "walking-distances-between-nodes.npz"
        self.modelled_volume_csv = pub / "pedestrian-volume-per-node.csv"
        self.volume_check_csv = pub / "pedestrian-volume-vs-counts.csv"
        self.map_layers_json = pub / "map-layers.json"                 # the downloaded files the maps draw

        # outputs/<run_id>/: the experiment
        self.run = run
        self.run_config = run / "config.yaml"                           # the settings the run used
        metric = self.metric_outputs = run / "metric" / cfg["metric_system"].split("/")[0]
        self.scores_csv = metric / "scores-per-usable-view.csv"
        self.all_views_csv = metric / "all-rated-views-with-inputs.csv"
        self.block_sections_csv = metric / "mean-score-per-block-section.csv"
        # the diagnostics
        self.pca_csv = metric / "dimension-pca-weights.csv"
        self.weight_robustness_csv = metric / "weight-robustness.csv"
        self.exponent_robustness_csv = metric / "exponent-robustness.csv"
        self.dimension_coherence_csv = metric / "dimension-coherence.csv"
        self.indicator_factors_csv = metric / "indicator-factor-loadings.csv"
        self.column_dictionary = metric / "column-dictionary.md"
        analysis = run / "analysis"
        check = self.vlm_rating_validation = analysis / "vlm-rating-validation"
        self.anchor_csv = check / "view-ratings-vs-segmentation.csv"
        self.enclosure_vs_geometry_csv = check / "enclosure-vs-canyon-geometry.csv"
        self.enclosure_curve_csv = check / "enclosure-vs-hw-curve.csv"
        self.street_seating_check_csv = check / "ratings-vs-street-seating.csv"
        self.cannot_judge_vs_vehicles_csv = check / "cannot-judge-vs-vehicle-share.csv"
        self.city_records_check_csv = check / "ratings-vs-city-records.csv"
        self.enclosure_by_height_csv = check / "enclosure-by-height.csv"
        self.physical_records_check_csv = check / "ratings-vs-physical-records.csv"  # agreement terms
        self.rating_repeatability_csv = check / "rating-repeatability.csv"            # every term
        self.footfall_regression_dir = analysis / "footfall-regression"
        self.model_fit_csv = self.footfall_regression_dir / "model-fit.csv"
        self.coefficients_csv = self.footfall_regression_dir / "regression-coefficients.csv"
        self.street_types = analysis / "street-type-clusters"
        self.clusters_csv = self.street_types / "cluster-per-view.csv"
        self.cluster_quality_csv = self.street_types / "cluster-quality.csv"
        self.centroids_csv = self.street_types / "cluster-centroids.csv"
        self.manova_csv = self.street_types / "manova-test.csv"
        self.tukey_csv = self.street_types / "tukey-pairwise-tests.csv"
        self.dwell_potential_dir = analysis / "dwell-potential"
        self.dwell_potential_nodes_csv = self.dwell_potential_dir / "dwell-potential-per-node.csv"
        self.dwell_potential_grid_csv = self.dwell_potential_dir / "dwell-potential-grid.csv"
        self.dwell_by_street_csv = self.dwell_potential_dir / "dwell-potential-per-street.csv"
        self.figures = analysis / "figures"
        # every result per node: for QGIS and as a table, what every column means, and the interactive viewer
        self.all_results_geojson = analysis / "all-results-per-node.geojson"
        self.all_results_csv = analysis / "all-results-per-node.csv"
        self.all_results_readme = analysis / "all-results-per-node.README.md"
        self.results_viewer_html = analysis / "results-viewer.html"

    def make_dirs(self):
        """!
        @brief Create every folder the run writes to.
        """
        for d in (self.data_prep_osm_cache, self.raw_photos, self.views, self.raw_model_outputs, self.fused_label_maps,
                  self.label_map_measurements, self.vlm_ratings, self.public_data_osm_cache, self.metric_outputs,
                  self.vlm_rating_validation, self.footfall_regression_dir, self.street_types, self.dwell_potential_dir,
                  self.figures):
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- what every step does first
def start(step, config=CONFIG):
    """!
    @brief Load the settings, create the folders, write the settings used to outputs/<run_id>/config.yaml.
    @return (cfg, paths).
    """
    cfg = load_config(config)
    paths = Paths(REPO, cfg)
    paths.make_dirs()
    paths.run_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"\n{'=' * 72}\n{step} — {cfg['name']} ({cfg['run_id']})\n{'=' * 72}")
    return cfg, paths
