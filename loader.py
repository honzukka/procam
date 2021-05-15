from __future__ import annotations
from argparse import Namespace
from typing import Optional, List

import torch
import numpy as np  # type: ignore

import utilities
from utilities import Metadata


class Loader:
    def __init__(self, config: Namespace):
        self.target_paths = config.target_paths
        self.model_path = config.model_path
        self.lt_matrix_paths = (
            config.lt_matrix_paths
            if config.lt_matrix_paths is not None else None
        )
        self.background_paths = (
            config.background_paths
            if config.background_paths is not None else None
        )

    def load(self, scene_ind: int) -> Data:
        target_imgs = [utilities.read_image(pth) for pth in self.target_paths]
        if self.model_path is not None:
            model = utilities.load_model(self.model_path)
        else:
            model = None
        if self.lt_matrix_paths is not None:
            lt_matrix_path = self.lt_matrix_paths[scene_ind]
            lt_matrix, extra_illum_img = utilities.load_matrix_wrap(
                lt_matrix_path
            )
            metadata = Metadata.create_from_matrix_file(
                lt_matrix_path
            )     # type: Optional[Metadata]
        else:
            lt_matrix = None
            metadata = None
            extra_illum_img = None
        if self.background_paths is not None:
            bg_img = utilities.read_image(self.background_paths[scene_ind])
        else:
            bg_img = None

        return Data(
            target_imgs, model, lt_matrix, metadata, bg_img, extra_illum_img
        )


class Data:
    def __init__(
        self, target_imgs: List[np.ndarray], model: Optional[torch.nn.Module],
        lt_matrix: Optional[np.ndarray],
        lt_matrix_metadata: Optional[Metadata],
        background_img: Optional[np.ndarray],
        extra_illum_img: Optional[np.ndarray]
    ):
        self.target_imgs = target_imgs
        self.model = model
        self.lt_matrix = lt_matrix
        self.lt_matrix_metadata = lt_matrix_metadata
        self.background_img = background_img
        self.extra_illum_img = extra_illum_img
