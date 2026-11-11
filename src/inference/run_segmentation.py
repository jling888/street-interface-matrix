"""!
@package run_segmentation
@brief Step run_segmentation_models (GPU): the three segmentation models (segmentation/, each a SegmentationModel) on
the photos:
       Mask2Former trained on Mapillary Vistas (the starting map), Mask2Former trained on ADE20K (facade detail) and
       SAM 3 (text-prompted detections, for classes neither has). fuse_segmentation_labels.py fuses their outputs (step
       fuse_label_maps).

In this order:
  1. the models   the three; the photos of the usable nodes; the device
  2. run_models   each model on every photo lacking its output for the current settings (resumable). Writes
                  data/<city>/inference/segmentation/<variant>/raw-model-outputs/
"""
import os

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from common.data_readers import photo
from inference.segmentation.mask2former import Mask2Former
from inference.segmentation.sam3 import SAM3


# ---------------------------------------------------------------- 1. the models
def segmentation_models(cfg, paths):
    """!
    @brief The three models (not loaded yet), by their role in fuse(), each with its folder under raw-model-outputs/
           named after the model (facebook/sam3 -> sam3/).
    """
    S = cfg["segmentation"]["models"]
    folder = lambda key: paths.raw_model_outputs / S[key].split("/")[-1]
    return {"mapillary": Mask2Former(cfg, folder("mapillary_model"), "mapillary"),
            "ade20k": Mask2Former(cfg, folder("ade20k_model"), "ade20k"),
            "sam3": SAM3(cfg, folder("sam3_model"))}


def usable_photos(paths):
    """!
    @brief The downloaded photos, without those of the nodes download marked unusable (user uploads and the like).
    @return street-view-frames.csv rows plus `name` (the photo's file stem).
    """
    manifest = pd.read_csv(paths.raw_photos_csv)
    meta = pd.read_csv(paths.pano_metadata_csv)        # reason: empty (NaN) = usable imagery
    skip = ~manifest.node_id.isin(meta.node_id[meta.reason.isna()])
    if skip.any():
        print(f"skipping {int(skip.sum())} photos of {manifest.node_id[skip].nunique()} nodes marked unusable "
              f"at download (e.g. user uploads)")
    manifest = manifest[~skip]
    return manifest.assign(name=[photo(n, h) for n, h in zip(manifest.node_id, manifest.heading)])


def device():
    """!
    @brief CUDA, then Apple MPS, then CPU.
    @return The torch device name.
    """
    import torch
    if torch.cuda.is_available():
        print(f"device: cuda -- {torch.cuda.get_device_name(0)}")
        return "cuda"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print("device: mps -- slow; a CUDA GPU is meant for full runs")
        return "mps"
    print("device: cpu -- very slow; a CUDA GPU is meant for full runs")
    return "cpu"


# ---------------------------------------------------------------- 2. run_models (GPU step)
def run_models(cfg, paths):
    """!
    @brief Step run_segmentation_models (GPU): the models on every photo still lacking an output the current settings
    need.
    """
    models = segmentation_models(cfg, paths).values()
    is_complete = lambda name: all(m.is_complete(name) for m in models)
    # what is still to do: a run cut short, or new settings, resumes from here
    manifest = usable_photos(paths)
    todo = manifest[[not is_complete(n) for n in manifest.name]]
    print(f"{len(manifest) - len(todo)} of {len(manifest)} photos have every raw output")
    if len(todo):
        print(f"{len(todo)} to run now")
        dev = device()
        for m in models:
            m.load_model(dev)
        failed = 0
        for r in tqdm(list(todo.itertuples()), desc="models", mininterval=5.0):
            try:
                img = Image.open(paths.raw_photos / r.file).convert("RGB")
            except OSError:
                failed += 1
                continue
            for m in models:                            # only what the photo lacks, saved as soon as it is done
                if not m.is_complete(r.name):
                    m.run_on_photo(img, r.name)
        print(f"ran the models on {len(todo) - failed} photos, {failed} would not open")
    # colab.sh watch stops at this line
    done = sum(is_complete(n) for n in manifest.name)
    print(f"run_segmentation_models done: {done} of {len(manifest)} photos have every raw output; "
          f"next: fuse_label_maps")
