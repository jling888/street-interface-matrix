"""!
@package base
@brief The base of every segmentation model in segmentation/: its raw output per photo, in the model's own classes
       (step run_segmentation_models, GPU), saved in a folder named after the model. run_segmentation.py runs the
       models;
       fuse_segmentation_labels.py maps their outputs to the taxonomy and fuses them (step fuse_label_maps).
"""
from abc import ABC, abstractmethod


class SegmentationModel(ABC):

    def __init__(self, folder):
        """!
        @param folder  The model's folder under raw-model-outputs/, named after the model: another model's outputs
                       are never taken for these.
        """
        self.folder = folder

    @abstractmethod
    def is_complete(self, name):
        """!
        @brief Whether a photo (its file stem) has every output the current settings need.
        """

    @abstractmethod
    def load_model(self, dev):
        """!
        @brief Load the model on the torch device `dev`.
        """

    @abstractmethod
    def run_on_photo(self, img, name):
        """!
        @brief Run the model on a photo (RGB) and save what is_complete() finds missing.
        """
