"""!
@package fuse_segmentation_labels
@brief Step fuse_label_maps (CPU, minutes): one label map per photo, fused from the three segmentation models' raw
outputs
       (run_segmentation.py, step run_segmentation_models), each read in the taxonomy of
       configs/inference/segmentation_classes.yaml.

In this order:
  1. fuse             one label per pixel: the Mapillary labels, refined class by class where the Mapillary label is one
                      the class may_replace; the class listed first wins an overlap
  2. fuse_label_maps  every photo whose raw outputs are complete. Writes data/<city>/inference/segmentation/<variant>/:
                      label-maps/<photo>.png, label-taxonomy.csv (id -> class), label-settings.json (the settings they
                      were fused with; check_settings)
Changing classes, priority, `may_replace`, sam3_min_scores (>= save_score_min) or gates needs only this step and the
ones after it.
"""
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from common.data_readers import IGNORE, class_taxonomy
from inference.run_segmentation import segmentation_models, usable_photos


# ---------------------------------------------------------------- 1. fuse
def fuse(cfg, mapillary, ade20k, detected):
    """!
    @brief One taxonomy label per pixel.

    Starts from the Mapillary labels. Each class then claims the pixels the ADE20K model or its prompted
    detections propose for it, wherever the Mapillary label there is one it may replace (its
    `may_replace`, default unknown only); a detected mask with less than the class's
    detect.min_share_on_replaceable on such pixels is dropped whole. Classes are applied from the bottom of
    segmentation_classes.yaml to the top, so where two classes claim the same pixel, the one listed first wins. Ignored
    pixels stay ignored.

    @param mapillary  HxW taxonomy ids from the Mapillary model.
    @param ade20k     HxW taxonomy ids from the ADE20K model.
    @param detected   {class: [HxW bool mask, ...]}, SAM 3's kept detections.
    @return HxW uint8 label map.
    """
    classes, tax = cfg["classes"], class_taxonomy(cfg)
    label = mapillary.astype(np.uint8)
    for c in reversed(list(classes)):
        cid = tax[c]
        may_replace = [tax[r] for r in classes[c].get("may_replace", ["unknown"])]
        allowed = np.isin(mapillary, may_replace + [cid])            # judged on the Mapillary label
        want = ade20k == cid
        for mask in detected.get(c, []):
            if (mask & allowed).sum() >= classes[c]["detect"].get("min_share_on_replaceable", 0) * mask.sum():
                want |= mask
        label[want & allowed] = cid
    return label


# ---------------------------------------------------------------- 2. fuse_label_maps
def settings(cfg):
    """!
    @brief The settings label maps are made with: the models and raw-output settings (segmentation.models), the
           fusion thresholds (segmentation.fusion) and the class table (segmentation_classes.yaml `classes`,
           `ignore`; saved under `fuse`, where they were before segmentation_classes.yaml, so label maps fused then
           still match). The terms belong to the label-map measurements, so changing them never invalidates label maps.
    @return dict, as saved in label-settings.json.
    """
    S = cfg["segmentation"]
    fusion = {**S["fusion"], "classes": cfg["classes"], "ignore": cfg["ignore"]}
    return json.loads(json.dumps({"segment": S["models"], "fuse": fusion}))


def check_settings(cfg, paths):
    """!
    @brief Stop if the label maps were fused with other settings (their class ids would be misread).
    """
    saved, maps = paths.fused_label_settings, paths.fused_label_maps
    if saved.exists() and maps.exists() and any(maps.iterdir()) and json.loads(saved.read_text()) != settings(cfg):
        sys.exit(f"segmentation settings changed since the label maps in {maps} were fused: run "
                 f"pipeline step fuse_label_maps (no GPU), or restore the settings saved in {saved}")


def fuse_label_maps(cfg, paths):
    """!
    @brief Step fuse_label_maps: one label map per photo whose raw outputs are complete.
    """
    models, tax = segmentation_models(cfg, paths), class_taxonomy(cfg)
    manifest = usable_photos(paths)
    ready = [n for n in manifest.name if all(m.is_complete(n) for m in models.values())]
    if len(ready) < len(manifest):
        print(f"{len(manifest) - len(ready)} photos lack raw outputs for the current settings "
              f"(run run_segmentation_models)")
    if not ready:
        print("nothing to fuse")
        return
    paths.fused_label_maps.mkdir(parents=True, exist_ok=True)
    for old in paths.fused_label_maps.glob("*.png"):    # label maps fused with other settings must not linger
        old.unlink()
    counts = np.zeros(256)
    for n in tqdm(ready, desc="fusing", mininterval=5.0):
        # the three raw outputs in the taxonomy (no model loaded), fused
        mapillary = models["mapillary"].read_classes(n)
        ade20k, detected = models["ade20k"].read_classes(n), models["sam3"].read_detections(n, mapillary.shape)
        label = fuse(cfg, mapillary, ade20k, detected)
        Image.fromarray(label).save(paths.fused_label_maps / f"{n}.png")
        counts += np.bincount(label.ravel(), minlength=256)
    # the settings the maps were fused with (the label-map measurements check them) and the id -> class key
    paths.fused_label_settings.write_text(json.dumps(settings(cfg), indent=1))
    pd.DataFrame({"id": list(tax.values()), "class": list(tax)}).to_csv(paths.fused_label_taxonomy, index=False)
    # a first look (the label-map measurements have the solid-angle-weighted shares per node)
    share = pd.Series({c: counts[i] for c, i in tax.items()}) / max(counts[:IGNORE].sum(), 1) * 100
    print("pixel share of the fused photos, %\n" + share.round(1).to_string())
    print(f"fuse_label_maps done: {len(ready)} of {len(manifest)} photos have a label map; "
          f"next: calculate_segmentation_pixel_shares")
