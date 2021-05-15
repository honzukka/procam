from __future__ import annotations
from typing import Tuple, List
from collections import OrderedDict
from argparse import Namespace

import torch
import numpy as np  # type: ignore

from modules import (
    Render, RenderSimple, GramLoss, L2Loss
)
import utilities
from loader import Data
import constants


class Model:
    def __init__(self, data: Data, config: Namespace, device: torch.device):
        self.data = data
        self.config = config
        self.device = device

        self.crop = self.get_crop()
        self.init_done = False

    def __iter__(self) -> Model:
        self.target_ind = 0
        return self

    def __next__(self) -> int:
        if self.target_ind >= len(self.data.target_imgs):
            raise StopIteration

        target_img = self.data.target_imgs[self.target_ind]
        self.set_target_and_assemble_model(target_img)
        self.init_done = True
        self.target_ind += 1
        return self.target_ind - 1

    # crop is either given by the user, or it is set to be the full image
    # (and thus has no effect when applied)
    def get_crop(self) -> List[int]:
        if (
            self.data.lt_matrix_metadata is None and
            self.data.background_img is None
        ):
            return None

        input_height, input_width = (
            (
                self.data.lt_matrix_metadata.image_height,
                self.data.lt_matrix_metadata.image_width
            )
            if self.data.lt_matrix_metadata is not None
            else (self.data.background_img.shape[0:2])
        )

        return utilities.get_crop(
            (
                self.config.crops[self.config.scene_ind]
                if self.config.crops is not None
                else None
            ),
            input_height, input_width
        )

    def set_target_and_assemble_model(self, target_img: np.ndarray):
        raise NotImplementedError()


class SynthesisModel(Model):
    '''
    Partial model that is initialized by random noise and uses Gram loss.
    '''
    def __init__(
        self, data: Data, config: Namespace, device: torch.device
    ):
        super(SynthesisModel, self).__init__(data, config, device)
        self.loss = GramLoss(data, config, self.crop).to(device)

    def __call__(
        self, img_and_brightness: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if not self.init_done:
            raise RuntimeError(
                'Model not iterated through before using! '
                '(__iter__() and __next__() not called)'
            )

        return self.model(img_and_brightness)

    def get_initial_guess(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.init_done:
            raise RuntimeError(
                'Model not iterated through before using! '
                '(__iter__() and __next__() not called)'
            )

        noise = (
            (
                torch.randn(
                    self.opt_size, dtype=torch.float32,
                    device=self.device
                ) * 0.01
            ) + self.loss.vgg_preprocessing.img_mean_bhwc_rgb
        ).clamp(constants.EPSILON, 1.0 - constants.EPSILON)

        brightness = torch.tensor(1, dtype=torch.float32, device=self.device)

        return noise, brightness


class TextureSynthesisModel(SynthesisModel):
    '''
    Final model that adds simple rendering (image-image multiplication)
    to a partial synthesis model.
    '''
    def __init__(self, data: Data, config: Namespace, device: torch.device):
        super(TextureSynthesisModel, self).__init__(data, config, device)

        # set opt_size (= size of the initial guess)
        if config.opt_size is not None:
            w, h = config.opt_size
            self.opt_size = (config.batch, h, w, 3)
            self.opt_size_given = True
        elif self.data.background_img is not None:
            self.opt_size = (config.batch, *self.data.background_img.shape)
            self.opt_size_given = True
        else:
            self.opt_size_given = False     # will be set based on target size

        self.render = RenderSimple(data, config).to(device)

    def set_target_and_assemble_model(self, target_image: np.ndarray):
        self.loss.update_target(target_image, self.device)

        # update opt_size if not fixed by the user
        if not self.opt_size_given:
            h, w, _ = target_image.shape
            self.opt_size = (self.config.batch, h, w, 3)

        # update crop if not deduced from background image
        if self.crop is None:
            self.loss.update_crop(
                utilities.get_crop(None, *self.opt_size[1:3])
            )

        # assemble the model
        self.model = torch.nn.Sequential(OrderedDict([
            ('render', self.render),
            ('loss', self.loss)
        ]))


class ProjectionSynthesisModel(SynthesisModel):
    '''
    Final model that adds full rendering (matrix-image multiplication)
    to a partial synthesis model.
    '''
    def __init__(self, data: Data, config: Namespace, device: torch.device):
        super(ProjectionSynthesisModel, self).__init__(data, config, device)

        # set opt_size according to projector resolution
        self.opt_size = (
            config.batch, data.lt_matrix_metadata.basis_height,
            data.lt_matrix_metadata.basis_width, 3
        )

        self.render = Render(data, config).to(device)

    def set_target_and_assemble_model(self, target_image: np.ndarray):
        assert self.crop is not None

        self.loss.update_target(target_image, self.device)

        # assemble the model
        self.model = torch.nn.Sequential(OrderedDict([
            ('render', self.render),
            ('loss', self.loss)
        ]))


class CompensationModel(Model):
    '''
    Partial model that is initialized by the target image and uses L2 loss.
    '''
    def __init__(self, data: Data, config: Namespace, device: torch.device):
        super(CompensationModel, self).__init__(data, config, device)
        self.loss = L2Loss(data, config, self.crop).to(device)

    def __call__(
        self, img_and_brightness: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if not self.init_done:
            raise RuntimeError(
                'Model not iterated through before using! '
                '(__iter__() and __next__() not called)'
            )

        return self.model(img_and_brightness)

    def get_initial_guess(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.init_done:
            raise RuntimeError(
                'Model not iterated through before using! '
                '(__iter__() and __next__() not called)'
            )

        brightness = torch.tensor(1, dtype=torch.float32, device=self.device)
        return self.initial_guess, brightness


class ProjectionCompensationModel(CompensationModel):
    '''
    Final model that adds full rendering (matrix-image multiplication)
    to a partial compensation model.
    '''
    def __init__(self, data: Data, config: Namespace, device: torch.device):
        super(ProjectionCompensationModel, self).__init__(data, config, device)

        metadata = data.lt_matrix_metadata
        self.basis_size = (metadata.basis_height, metadata.basis_width)

        self.render = Render(data, config).to(device)

    def set_target_and_assemble_model(self, target_img: np.ndarray):
        assert self.crop is not None

        # turn the target into the initial guess
        self.initial_guess = torch.from_numpy(
            utilities.resize(target_img, self.basis_size, 'lanczos')
        )[None, :, :, :].to(self.device)

        target_img = torch.from_numpy(
            target_img
        )[None, :, :, :].to(self.device)
        self.loss.update_target(target_img)

        self.model = torch.nn.Sequential(OrderedDict([
            ('render', self.render),
            ('loss', self.loss)
        ]))


class TextureCompensationModel(CompensationModel):
    '''
    Final model that adds simple rendering (image-image multiplication)
    to a partial compensation model.
    '''
    def __init__(self, data: Data, config: Namespace, device: torch.device):
        super(TextureCompensationModel, self).__init__(data, config, device)

        # no point in compensating without a background image...
        if data.background_img is None:
            raise ValueError('No background image provided.')
        self.bg_height, self.bg_width, _ = data.background_img.shape

        self.render = RenderSimple(data, config).to(device)

    def set_target_and_assemble_model(self, target_img: np.ndarray):
        if self.crop is None:
            self.loss.update_crop(
                utilities.get_crop(None, self.bg_height, self.bg_width)
            )

        target_img = utilities.resize(
            target_img, (self.bg_height, self.bg_width), 'lanczos'
        )
        target_img = torch.from_numpy(
            target_img
        )[None, :, :, :].to(self.device)
        self.initial_guess = target_img.clone()

        self.loss.update_target(target_img)

        self.model = torch.nn.Sequential(OrderedDict([
            ('render', self.render),
            ('loss', self.loss)
        ]))
