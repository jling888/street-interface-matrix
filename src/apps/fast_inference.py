"""!
@package fast_inference
@brief Segment any folder of images with the study's models and fusion, outside the pipeline: no study area, nodes or
       views, so the label maps of any image match the study's. Not a pipeline step (docs/apps.md):

    src/apps/fast_inference.sh <image folder> [--out <folder>] [--exp <experiment>] [--limit N]

In this order:
  1. images  every image under the folder (any depth), named by its path there, '/' as '__'
  2. run     the three models (run_segmentation.py) on each image still lacking a label map, fused (fuse()); an
             image that cannot be read or run is logged and skipped. Writes <out>/ (default <image folder>-segmented/):
             raw-model-outputs/ (each model's output, so a run cut short resumes), label-maps/<image>.png and
             label-taxonomy.csv (id -> class), overlays/<image>.png, class-shares.csv (one row per image: each
             class's share of the pixels not ignored, %), errors.csv, label-settings.json (another run into the folder
             must use the same settings)
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # src/
from apps.validation_app import overlay
from common.data_readers import IGNORE, class_taxonomy
from common.settings_and_paths import CONFIG, CONFIGS, load_config
from inference.fuse_segmentation_labels import fuse, settings
from inference.run_segmentation import device, segmentation_models

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------- 1. images
def images(folder, limit=None):
    """!
    @return {image name: path}, every image under `folder` in path order, the first `limit` if given.
    """
    found = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGES)[:limit]
    return {"__".join(p.relative_to(folder).with_suffix("").parts): p for p in found}


def class_shares(label, tax):
    """!
    @return {<class>_pct: its share of the pixels not ignored, %}.
    """
    n = np.bincount(label.ravel(), minlength=256)
    valid = max(n[:IGNORE].sum(), 1)
    return {f"{c}_pct": 100 * n[i] / valid for c, i in tax.items()}


# ---------------------------------------------------------------- 2. run
def run(folder, out, cfg, limit=None):
    """!
    @brief Label maps, overlays and class shares for every image under `folder` (see the module).
    """
    maps, overlays = out / "label-maps", out / "overlays"
    for d in (maps, overlays):
        d.mkdir(parents=True, exist_ok=True)
    saved = out / "label-settings.json"
    if saved.exists() and json.loads(saved.read_text()) != settings(cfg):
        sys.exit(f"{out} was made with other segmentation settings: give another --out")
    saved.write_text(json.dumps(settings(cfg), indent=1))
    tax = class_taxonomy(cfg)
    pd.DataFrame({"id": list(tax.values()), "class": list(tax)}).to_csv(out / "label-taxonomy.csv", index=False)

    found = images(folder, limit)
    todo = {n: p for n, p in found.items() if not (maps / f"{n}.png").exists()}
    print(f"{len(found)} images, {len(found) - len(todo)} already done, {len(todo)} to run")
    models = segmentation_models(cfg, SimpleNamespace(raw_model_outputs=out / "raw-model-outputs"))
    errors = []
    if todo:
        dev = device()
        for m in models.values():
            m.load_model(dev)
        for n, p in tqdm(todo.items(), desc="images", mininterval=5.0):
            try:
                img = Image.open(p).convert("RGB")
                for m in models.values():               # only what the image lacks, saved as soon as it is done
                    if not m.is_complete(n):
                        m.run_on_photo(img, n)
                mapillary = models["mapillary"].read_classes(n)
                label = fuse(cfg, mapillary, models["ade20k"].read_classes(n),
                             models["sam3"].read_detections(n, mapillary.shape))
            except Exception as ex:                     # one bad image never stops the batch
                errors.append({"image": n, "file": str(p), "error": repr(ex)})
                continue
            Image.fromarray(label).save(maps / f"{n}.png")
            Image.fromarray(overlay(p, maps / f"{n}.png")).save(overlays / f"{n}.png")

    # every label map in the folder, this run's and earlier ones'
    rows = [{"image": n, "file": str(p), **class_shares(np.asarray(Image.open(maps / f"{n}.png")), tax)}
            for n, p in found.items() if (maps / f"{n}.png").exists()]
    pd.DataFrame(rows).to_csv(out / "class-shares.csv", index=False)
    pd.DataFrame(errors, columns=["image", "file", "error"]).to_csv(out / "errors.csv", index=False)
    print(f"fast_inference done: {len(rows)} of {len(found)} images have a label map, {len(errors)} failed; {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Segment any folder of images with the study's models (docs/apps.md).")
    ap.add_argument("folder", type=Path, help="the images, at any depth")
    ap.add_argument("--out", type=Path, help="the output folder (default <folder>-segmented)")
    ap.add_argument("--exp", help="the experiment whose segmentation settings to use (default the study's)")
    ap.add_argument("--limit", type=int, help="only the first N images: a quick test")
    a = ap.parse_args()
    if not a.folder.is_dir():
        sys.exit(f"no folder {a.folder}")
    folder = a.folder.resolve()
    run(folder, a.out or folder.with_name(folder.name + "-segmented"),
        load_config(CONFIGS / "experiments" / f"{a.exp}.yaml" if a.exp else CONFIG), a.limit)
