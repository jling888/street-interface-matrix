"""!
@package mask2former
@brief Mask2Former, one label per pixel, trained on Mapillary Vistas (the starting map) or on ADE20K (facade detail):
       the model's own label ids per photo (step run_segmentation_models), read back in the taxonomy through the label
       names
       segmentation_classes.yaml lists for it (step fuse_label_maps).

Writes <folder>/: <photo>.png and label-names.json (the model's id -> name, so fusing needs no model).
"""
import json
import sys
from functools import cached_property

import numpy as np
from PIL import Image

from common.data_readers import IGNORE, class_taxonomy
from inference.segmentation.base import SegmentationModel


class Mask2Former(SegmentationModel):

    def __init__(self, cfg, folder, dataset):
        """!
        @param dataset  "mapillary" or "ade20k": its keys in the config (<dataset>_model) and in
                        segmentation_classes.yaml.
        """
        super().__init__(folder)
        self.dataset, self.model_name = dataset, cfg["segmentation"]["models"][f"{dataset}_model"]
        self.classes, self.ignore, self.tax = cfg["classes"], cfg["ignore"], class_taxonomy(cfg)

    def labels_path(self, name):
        return self.folder / f"{name}.png"

    def is_complete(self, name):
        return self.labels_path(name).exists()

    def load_model(self, dev):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.proc = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name).to(dev).eval()
        self.dev = dev
        id2label = self.model.config.id2label
        if max(map(int, id2label)) >= IGNORE:          # 255 is fusion's ignore id
            sys.exit(f"segmentation.models.{self.dataset}_model has more labels than a uint8 label map can hold")
        self.folder.mkdir(parents=True, exist_ok=True)
        label_names = {int(i): lab for i, lab in id2label.items()}
        (self.folder / "label-names.json").write_text(json.dumps(label_names, indent=1))

    def run_on_photo(self, img, name):
        """!
        @brief The model's own label id per pixel, saved as a PNG.
        """
        import torch
        inp = self.proc(images=[img], return_tensors="pt").to(self.dev)
        with torch.inference_mode():
            out = self.model(**inp)
        out.class_queries_logits = out.class_queries_logits.cpu().contiguous()   # MPS returns non-contiguous
        out.masks_queries_logits = out.masks_queries_logits.cpu().contiguous()
        labels = self.proc.post_process_semantic_segmentation(out, target_sizes=[img.size[::-1]])[0]
        Image.fromarray(labels.numpy().astype(np.uint8)).save(self.labels_path(name))

    def build_lookup_table(self, id2label):
        """!
        @brief A lookup table from the model's label ids to taxonomy ids.
        @param id2label  The model's {id: label name}.
        @return uint8 array indexed by the model's label id: taxonomy id; 0 where unmapped; IGNORE for the labels in
                segmentation_classes.yaml ignore. Exits if segmentation_classes.yaml names a label the model does not
                have.
        """
        names = {lab: int(i) for i, lab in id2label.items()}
        wanted = [(self.tax[c], lab) for c, spec in self.classes.items() for lab in spec.get(self.dataset, [])]
        wanted += [(IGNORE, lab) for lab in (self.ignore or {}).get(self.dataset, [])]
        absent = sorted(lab for _, lab in wanted if lab not in names)
        if absent:
            sys.exit(f"segmentation_classes.yaml names {self.dataset} labels the model does not have: {absent}")
        lut = np.zeros(256, dtype=np.uint8)
        for tid, lab in wanted:
            lut[names[lab]] = tid
        return lut

    @cached_property
    def lookup_table(self):
        return self.build_lookup_table(json.loads((self.folder / "label-names.json").read_text()))

    def read_classes(self, name):
        """!
        @brief A photo's saved labels in the taxonomy.
        @return HxW uint8 taxonomy ids.
        """
        return self.lookup_table[np.asarray(Image.open(self.labels_path(name)))]
