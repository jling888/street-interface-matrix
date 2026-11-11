"""!
@package sam3
@brief SAM 3, which finds and masks every match of a text prompt: per photo and detector pass (one class's prompts,
       segmentation_classes.yaml detect.prompts, on one tile) every detection scoring at least save_score_min, with
       its mask (step run_segmentation_models); read back as each class's detections kept by its thresholds and gates
       (step fuse_label_maps).

The model never sees segmentation_classes.yaml except the prompts, so only a new or changed prompt list (or
save_score_min or the model) needs step run_segmentation_models again, and then only for the passes it changes. Writes
<folder>/<pass hash>/<photo>.npz: the pass's boxes, scores and packed masks.
"""
import hashlib
import json
import os
import sys

import numpy as np

from inference.segmentation.base import SegmentationModel

WHOLE = [[0, 0, 1, 1]]                                           # the default tile: the whole photo


def crop_tile(img, tile):
    """!
    @brief One tile of the photo.
    @param tile  [x0, y0, x1, y1] as fractions of the photo.
    @return (the cropped image, its left and top offset in photo pixels).
    """
    W, H = img.size
    fx0, fy0, fx1, fy1 = tile
    ox, oy = int(fx0 * W), int(fy0 * H)
    return img.crop((ox, oy, int(fx1 * W), int(fy1 * H))), ox, oy


def encode_tile(proc, model, img, tile, dev):
    """!
    @brief SAM 3's encoding of one tile of the photo: the expensive step, done once per tile and shared by
           every prompt that searches it.
    @return (crop, its left and top offset in photo pixels, the encoding).
    """
    import torch
    crop, ox, oy = crop_tile(img, tile)
    pixels = proc(images=crop, return_tensors="pt")["pixel_values"].to(dev)
    with torch.inference_mode():
        return crop, ox, oy, model.get_vision_features(pixel_values=pixels)


def run_prompts_on_tile(proc, model, encoded, prompts, dev, score_min):
    """!
    @brief One SAM 3 pass: every match of each of one class's prompts on one encoded tile.
    @param encoded    encode_tile() of the tile.
    @param score_min  segmentation.models.save_score_min.
    @return (boxes as [[x0, y0, x1, y1], ...] in photo pixels, scores, (n, h, w) bool masks of the tile,
            the tile's offset (x, y)); every detection scoring >= score_min. A thing two prompts both
            match appears twice; fusion unites them.
    """
    import torch
    crop, ox, oy, vision = encoded
    boxes, scores, masks = [], [], []
    for prompt in prompts:
        text = proc(text=prompt, return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model(vision_embeds=vision, input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        found = proc.post_process_instance_segmentation(out, threshold=score_min, mask_threshold=0.5,
                                                        target_sizes=[(crop.height, crop.width)])[0]
        for (x0, y0, x1, y1), sc, mk in zip(found["boxes"].tolist(), found["scores"].tolist(), found["masks"].cpu().numpy()):
            boxes.append([x0 + ox, y0 + oy, x1 + ox, y1 + oy])
            scores.append(sc)
            masks.append(mk > 0)
    return boxes, scores, np.array(masks, bool).reshape(-1, crop.height, crop.width), (ox, oy)


def pack_masks(masks, offset=(0, 0)):
    """!
    @brief Bool masks, stored compactly: each cropped to its own bounding box, bit-packed.
    @param masks   (n, h, w) bool masks, of the whole photo or of a tile.
    @param offset  (x, y) of the masks' top-left corner in the photo (a tile's offset; (0, 0) for the photo).
    @return {"bbox": (n, 4) int32 [x0, y0, x1, y1) in photo pixels, "offs": (n + 1,) offsets into "bits",
            "bits": uint8}.
    """
    ox, oy = offset
    bbox, chunks = [], []
    for mk in masks:
        ys, xs = np.nonzero(mk)
        b = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [0, 0, 0, 0]
        chunks.append(np.packbits(mk[b[1]:b[3], b[0]:b[2]].ravel()))
        bbox.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy] if len(xs) else b)
    return {"bbox": np.array(bbox, np.int32).reshape(-1, 4), "offs": np.cumsum([0] + [len(c) for c in chunks]),
            "bits": np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)}


def unpack_mask(bbox, offs, bits, i, shape):
    """!
    @brief Mask i of a pack_masks(), back at full size.
    @return HxW bool array.
    """
    x0, y0, x1, y1 = bbox[i]
    mk = np.zeros(shape, bool)
    n = (x1 - x0) * (y1 - y0)
    if n:
        mk[y0:y1, x0:x1] = np.unpackbits(bits[offs[i]:offs[i + 1]])[:n].reshape(y1 - y0, x1 - x0).astype(bool)
    return mk


class SAM3(SegmentationModel):

    def __init__(self, cfg, folder):
        super().__init__(folder)
        self.S = cfg["segmentation"]["models"]
        self.F = cfg["segmentation"]["fusion"]                 # the score thresholds and gates
        self.prompted_classes = {c: spec for c, spec in cfg["classes"].items() if spec.get("detect")}
        low = {c: self.min_score(spec) for c, spec in self.prompted_classes.items()
               if self.min_score(spec) < self.S["save_score_min"]}
        if low:
            sys.exit(f"segmentation_classes.yaml: detect.sam3_min_score below segmentation.models.save_score_min "
                     f"{self.S['save_score_min']} (detections that low are never saved): {low}")
        # every pass the settings need, {its folder: (prompts, tile)}; classes with the same prompts and tile share one
        self.needed_passes = {self.pass_dir(spec, t): (spec["detect"]["prompts"], t)
                              for spec in self.prompted_classes.values() for t in spec["detect"].get("tiles", WHOLE)}

    def min_score(self, spec):
        """!
        @brief A class's lowest counting score.
        @param spec  The class's entry in segmentation_classes.yaml.
        """
        return spec["detect"].get("sam3_min_score", self.F["sam3_min_score"])

    def pass_dir(self, spec, tile):
        """!
        @brief The folder of one detector pass, named by the hash of all that shapes its output: a changed prompt list,
               tile or save_score_min gets a new folder, so only that pass is run again.
        """
        what = ["sam3", self.S["sam3_model"], self.S["save_score_min"], spec["detect"]["prompts"], tile]
        return self.folder / hashlib.sha1(json.dumps(what).encode()).hexdigest()[:10]

    def is_complete(self, name):
        return all((f / f"{name}.npz").exists() for f in self.needed_passes)

    def load_model(self, dev):
        from transformers import Sam3Model, Sam3Processor
        name = self.S["sam3_model"]
        self.proc, self.dev = Sam3Processor.from_pretrained(name), dev
        self.model = Sam3Model.from_pretrained(name).to(dev).eval()
        for f in self.needed_passes:
            f.mkdir(parents=True, exist_ok=True)

    def run_on_photo(self, img, name):
        """!
        @brief The passes a photo lacks, each saved in its folder.
        """
        encoded = {}                                    # each tile encoded once, for every prompt that searches it
        for folder, (prompts, tile) in self.needed_passes.items():
            out = folder / f"{name}.npz"
            if out.exists():
                continue
            if tuple(tile) not in encoded:
                encoded[tuple(tile)] = encode_tile(self.proc, self.model, img, tile, self.dev)
            boxes, scores, masks, offset = run_prompts_on_tile(self.proc, self.model, encoded[tuple(tile)], prompts,
                                                               self.dev, self.S["save_score_min"])
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez_compressed(tmp, boxes=np.array(boxes, np.float32).reshape(-1, 4),
                                scores=np.array(scores, np.float32), **pack_masks(masks, offset))
            os.replace(tmp, out)                        # never a half-written file, even if the run is killed

    def kept_detections(self, spec, name, shape):
        """!
        @brief One class's detections in a photo, kept by its settings.

        A detection is kept if its score reaches the class's detect.sam3_min_score (fusion.sam3_min_score by default)
        and its box covers at most max_box_share of the photo; with detect.street_level it must also reach below
        the horizon (the photos are level, so the middle row) and be at most street_level_max_height of the
        photo tall. A mask larger than the class's detect.max_mask_share is dropped.

        @param spec   The class's entry in segmentation_classes.yaml.
        @param shape  The photo's (H, W).
        @return [HxW bool mask, ...].
        """
        F, d = self.F, spec["detect"]
        H, W = shape
        out = []
        for tile in d.get("tiles", WHOLE):
            with np.load(self.pass_dir(spec, tile) / f"{name}.npz") as z:
                boxes, scores, bbox, offs, bits = (z[f] for f in ("boxes", "scores", "bbox", "offs", "bits"))
            for i, ((x0, y0, x1, y1), s) in enumerate(zip(boxes, scores)):
                if s < self.min_score(spec) or (x1 - x0) * (y1 - y0) / (W * H) > F["max_box_share"]:
                    continue
                if d.get("street_level") and (y1 < H / 2 or (y1 - y0) / H > F["street_level_max_height"]):
                    continue                            # must reach below the horizon, and not be tower-sized
                mk = unpack_mask(bbox, offs, bits, i, shape)
                if mk.mean() <= d.get("max_mask_share", 1.0):
                    out.append(mk)
        return out

    def read_detections(self, name, shape):
        """!
        @brief A photo's kept detections per class.
        @return {class: [HxW bool mask, ...]}.
        """
        return {c: self.kept_detections(spec, name, shape) for c, spec in self.prompted_classes.items()}
