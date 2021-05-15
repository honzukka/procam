from __future__ import annotations
from typing import List, Tuple
from argparse import Namespace
from collections import OrderedDict

import torch
import torch.nn.functional as F

import utilities
import lanczos
from loader import Data
import constants


class RenderSimple(torch.nn.Module):
    '''
    Performs element-wise multiplication of an input image with a background
    image and scales the result by base brightness and input brightness.
    '''
    def __init__(self, data: Data, config: Namespace):
        super(RenderSimple, self).__init__()
        if data.background_img is not None:
            self.background_img = torch.from_numpy(
                data.background_img
            )[None, :, :, :]
            self.render_f = self.render_with_background
        else:
            self.render_f = self.render_without_background
        self.base_brightness = config.base_brightness

    def forward(
        self, img_and_brightness: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return self.render_f(*img_and_brightness)

    def render_with_background(
        self, x: torch.Tensor, brightness: torch.Tensor
    ) -> torch.Tensor:
        if x.shape[1:] != self.background_img.shape[1:]:
            x = F.interpolate(
                x.permute(0, 3, 1, 2),
                self.background_img.shape[1:3],
                mode='bilinear'
            ).permute(0, 2, 3, 1)
        return x * self.background_img * self.base_brightness * brightness

    def render_without_background(
        self, x: torch.Tensor, brightness: torch.Tensor
    ) -> torch.Tensor:
        return x * self.base_brightness * brightness

    # used for .to(device) calls
    def _apply(self, fn):
        super(RenderSimple, self)._apply(fn)
        if hasattr(self, 'background_img'):
            self.background_img = fn(self.background_img)
        return self


class Render(torch.nn.Module):
    '''
    Performs multiplication of the light transport (LT) matrix and
    the input image (which effectively projects the input image onto a scene).
    The result is scaled by base brightness and input brightness.
    There is also the option to perform this on multiple GPUs if the
    LT matrix is too large.
    '''
    def __init__(self, data: Data, config: Namespace):
        super(Render, self).__init__()
        if config.n_gpus <= 1:
            self.lt_matrix = torch.from_numpy(data.lt_matrix)
        else:
            self.lt_matrix_chunks = self.split_across_gpus(
                data.lt_matrix, config.n_gpus
            )

        self.extra_illum = (
            torch.from_numpy(data.extra_illum_img)[None, :, :, :]
            if data.extra_illum_img is not None
            else torch.zeros((1, *data.lt_matrix.shape[:3]))
        )
        self.base_brightness = config.base_brightness

    def forward(
        self, img_and_brightness: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        img, brightness = img_and_brightness
        n, hb, wb, c = img.shape
        img_reshaped = img.transpose(1, 2).reshape(n, hb * wb, 3)

        if hasattr(self, 'lt_matrix'):
            render = self.render_single_gpu(img_reshaped)
        else:
            render = self.render_multi_gpu(img_reshaped)

        return (
            (render * self.base_brightness * brightness) + self.extra_illum
        )

    def render_single_gpu(self, img: torch.Tensor):
        return torch.einsum('hwcb,nbc->nhwc', self.lt_matrix, img)

    def render_multi_gpu(self, img: torch.Tensor):
        b = img.shape[0]
        render = torch.zeros(
            (b, *self.lt_matrix_chunks[0].shape[0:3]), dtype=torch.float32
        ).cuda(0)
        start = 0
        for i, chunk in enumerate(self.lt_matrix_chunks):
            chunk_size = chunk.shape[3]
            img_chunk_cuda = img[:, start:start + chunk_size, :].cuda(i)
            render += torch.einsum(
                'hwcb,nbc->nhwc', chunk, img_chunk_cuda
            ).cuda(0)
            start += chunk_size
        return render

    def split_across_gpus(self, lt_matrix, n_gpus):
        basic_chunk_size = int(lt_matrix.shape[3] / n_gpus)
        remainder = lt_matrix.shape[3] % n_gpus
        lt_matrix_chunks = []
        start = 0
        for i in range(n_gpus):
            chunk_size = (
                basic_chunk_size + 1
                if i < remainder else basic_chunk_size
            )
            lt_matrix_chunks.append(
                torch.from_numpy(
                    lt_matrix[:, :, :, start:start+chunk_size]
                ).cuda(i)
            )
            start += chunk_size
        assert start == lt_matrix.shape[3]
        return lt_matrix_chunks

    # used for .to(device) calls
    def _apply(self, fn):
        super(Render, self)._apply(fn)
        if hasattr(self, 'lt_matrix'):
            self.lt_matrix = fn(self.lt_matrix)
        self.extra_illum = fn(self.extra_illum)
        return self


class L2Loss(torch.nn.Module):
    '''
    Crops the input image and returns its L2 distance from the target image.
    '''
    def __init__(
        self, data: Data, config: Namespace, crop: List[int]
    ):
        super(L2Loss, self).__init__()
        self.crop = crop

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        cropped_img = (
            image[:, self.crop[1]:self.crop[3], self.crop[0]:self.crop[2], :]
        )
        resized_img = F.interpolate(
            cropped_img.permute(0, 3, 1, 2),
            self.target_image.shape[1:3],
            mode='bilinear'
        ).permute(0, 2, 3, 1)
        return F.mse_loss(resized_img, self.target_image, reduction='mean')

    def update_target(self, target_image: torch.Tensor):
        self.target_image = target_image

    def update_crop(self, crop: List[int]):
        self.crop = crop

    # used for .to(device) calls
    def _apply(self, fn):
        super(L2Loss, self)._apply(fn)
        if hasattr(self, 'target_image'):
            self.target_image = fn(self.target_image)
        return self


class GramLoss(torch.nn.Module):
    '''
    Crops the input image and returns the Gram loss with respect to the target.
    Based on Gatys et al. (https://arxiv.org/abs/1505.07376).
    Contains layer-chaining by Novak et al. (https://arxiv.org/abs/1605.04603)
    and multi-scale Gaussian pyramid by Snelgrove
    (https://dl.acm.org/doi/pdf/10.1145/3145749.3149449)
    '''
    def __init__(
        self, data: Data, config: Namespace, crop: List[int]
    ):
        super(GramLoss, self).__init__()
        self.crop = crop

        self.batch = config.batch
        self.loss_layers = config.loss_layers
        self.offset = config.gram_offset
        self.layer_weights = config.layer_weights
        self.pyramid_weights = config.pyramid_weights

        self.pyramid = Pyramid(config.pyramid_layers)
        self.vgg_preprocessing = VGGPreprocessing()
        self.vgg = VGGWrapper(data.model, config.layer_names)

        self.model = torch.nn.Sequential(OrderedDict([
            ('pyramid', self.pyramid),
            ('vgg_preprocessing', self.vgg_preprocessing),
            ('vgg', self.vgg)
        ]))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        cropped_img = (
            img[:, self.crop[1]:self.crop[3], self.crop[0]:self.crop[2], :]
        )
        activations = self.model(cropped_img)
        gram_matrices = self.compute_gram_matrices(activations)

        # compute loss
        loss = torch.tensor(0.0, dtype=torch.float32, device=img.device)
        for i, pyramid_gram_matrices in enumerate(gram_matrices):
            pyramid_weight = self.pyramid_weights[i]
            pyramid_loss = torch.zeros_like(loss)
            for j, gram in enumerate(pyramid_gram_matrices):
                target_gram = self.target_gram_matrices[i][j]
                layer_weight = self.layer_weights[j]
                pyramid_loss += (
                    layer_weight *
                    F.mse_loss(gram, target_gram, reduction='sum')
                )
            loss += pyramid_weight * pyramid_loss

        return loss

    def update_target(self, target_image: torch.Tensor, device: torch.device):
        # extract VGG activations of the target image
        target_image = (
            torch.from_numpy(target_image)
            .expand(self.batch, *target_image.shape)
            .to(device)
        )
        activations = self.model(target_image)

        # detach activations from the computation graph
        for i, _ in enumerate(activations):
            for j, _ in enumerate(activations[i]):
                activations[i][j] = activations[i][j].detach()

        # compute Gram matrices of all target activations (pyramid, chains...)
        self.target_gram_matrices = self.compute_gram_matrices(activations)

    def update_crop(self, crop: List[int]):
        self.crop = crop

    def compute_gram_matrices(self, activations):
        gram_matrices = []
        for pyramid_activations in activations:
            pyramid_gram_matrices = []
            for a, b in self.loss_layers:
                # bring the activations to the same size if one is smaller
                left = pyramid_activations[a if a <= b else b]
                right = pyramid_activations[b if a <= b else a]
                right_up = F.interpolate(right, left.shape[2:], mode='nearest')
                gram = utilities.gram_matrix(left, right_up, self.offset)
                pyramid_gram_matrices.append(gram)
            gram_matrices.append(pyramid_gram_matrices)
        return gram_matrices

    # used for .to(device) calls
    def _apply(self, fn):
        super(GramLoss, self)._apply(fn)
        self.model._apply(fn)
        return self


class Pyramid(torch.nn.Module):
    def __init__(self, layers: int):
        super(Pyramid, self).__init__()
        self.layers = layers

    def forward(self, img: torch.Tensor) -> List[torch.Tensor]:
        img_pyramid = [img]

        # TODO: add a scale limit, so that images don't get too small
        for i in range(1, self.layers):
            img_pyramid.append(lanczos.downsample(img, scale=2**i))

        return img_pyramid


class VGGPreprocessing(torch.nn.Module):
    def __init__(self):
        super(VGGPreprocessing, self).__init__()
        self.img_mean_bhwc_rgb = torch.tensor(
            constants.DEFAULT_IMAGE_MEAN_RGB, dtype=torch.float32
        )[None, None, None, :]

    def forward(self, img_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        # center, rescale, to BCHW and to BGR
        for i in range(len(img_pyramid)):
            img_scaled = (img_pyramid[i] - self.img_mean_bhwc_rgb) * 255
            img_pyramid[i] = img_scaled.permute(0, 3, 1, 2).flip([1])
        return img_pyramid

    # used for .to(device) calls
    def _apply(self, fn):
        super(VGGPreprocessing, self)._apply(fn)
        self.img_mean_bhwc_rgb = fn(self.img_mean_bhwc_rgb)
        return self


class VGGWrapper(torch.nn.Module):
    def __init__(self, vgg: torch.nn.Module, layer_names: List[str]):
        super(VGGWrapper, self).__init__()
        self.layer_names = layer_names

        # remove unnecessary layers to speed up the forward pass
        i = 0
        for name, layer in vgg.named_children():
            if name == layer_names[-1]:
                break
            i += 1
        self.vgg = vgg[:(i + 1)].eval()

        # insert hook to store activations on requested layers
        self.hook = ActivationsHook()
        for name, layer in self.vgg.named_children():
            if name in layer_names:
                layer.register_forward_hook(self.hook)

    def forward(
        self, img_pyramid: List[torch.Tensor]
    ) -> List[List[torch.Tensor]]:
        activations = []
        for img in img_pyramid:
            self.hook.clear()
            self.vgg(img)
            activations.append(self.hook.activations)
        return activations

    # used for .to(device) calls
    def _apply(self, fn):
        super(VGGWrapper, self)._apply(fn)
        self.vgg._apply(fn)
        return self


class ActivationsHook:
    def __init__(self):
        self.activations = []

    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        self.activations.append(layer_out)

    def clear(self):
        self.activations = []
