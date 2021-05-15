import argparse
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np      # type: ignore

import utilities


def main():
    args = parse_arguments()
    torch.device('cpu')

    image = torch.from_numpy(utilities.read_image(args.img_path))
    downsampled_image = downsample(image, scale=4)

    success = utilities.write_image(
        'output.exr',
        np.ascontiguousarray(downsampled_image.numpy()),
        np.float32
    )

    print('Downsampled image successfully saved: {}'.format(success))


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'img_path',
        help='Input image path.'
    )

    return parser.parse_args()


def scale_to_target(
    src: torch.Tensor, target_res: Tuple[int, int],
    pad_color: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    '''
    lanczos downsample an image to be a certain size. Keeps the aspect ration,
    uses integer scales and pad if necessary (almost always...)
    '''
    assert type(src) == torch.Tensor

    if len(src.shape) == 3:
        src = src[None, :, :, :]

    assert len(src.shape) == 4

    # downsample if target resolution is smaller than source resolution
    frames, src_height, src_width, _ = src.shape
    scale = 1
    scale = max(scale, math.ceil(src_height / target_res[0]))
    scale = max(scale, math.ceil(src_width / target_res[1]))
    downsampled = downsample(src, scale=scale) if scale > 1 else src

    # now src has lower resolution than target, so pad
    target = torch.ones(
        (frames, target_res[0], target_res[1], 3),
        device=device, dtype=torch.float32
    ) * pad_color
    target[:, 0:downsampled.shape[1], 0:downsampled.shape[2], :] = downsampled

    return target


def downsample(
    image: torch.Tensor, a: int = 2, scale: int = 2
) -> torch.Tensor:
    '''
    lanczos downsample an image; image assumed to be in format
    ([B], H, W, C) where B is optional
    '''
    # the main idea is to avoid aliasing when downsampling

    # ideally, this can be done by converting the image to frequency
    # domain and multiplying by a box function to cut off frequencies
    # that are too high for the new sampling rate

    # in spatial domain, the equivalent of this is convolving the image
    # with a sinc function

    # let's do it the Lanczos way!

    kernel = build_kernel(a, scale, image.shape[-1], image.device)
    padded_image = pad_image(image, a, scale)

    result = F.conv2d(
        padded_image, kernel, stride=scale,
        groups=image.shape[-1]
    ).permute(0, 2, 3, 1)

    # remove batch dimension if necessary
    return result if len(image.shape) == 4 else result[0]


def build_kernel(
    a: int = 2, scale: int = 2, n_channels: int = 3,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    # we have 2 problems now:
    #   1) sinc is unbounded
    #   2) bounding the sinc introduces ringing artefacts
    # that's where Lanczos comes to rescue!

    # the Lanczos kernel represents a windowed sinc function
    # a:     the number of sinc lobes that we wish to include before we cut off
    # scale: the extent to which we want to eliminate high frequencies

    # first, we set the extent (2 * radius) of the kernel
    # scale is included here because it makes the function wider,
    # so we need to cut off further to make sure
    # we keep the correct number of lobes
    # we shift by 0.5 because sinc(x) = 0 for integral |x| > 0
    x = torch.linspace(
        -scale * a + 0.5,
        scale * a - 0.5,
        2 * scale * a,
        device=device
    )
    pi = 3.1416

    wider_sinc = sinc((pi * x) / scale)
    lanczos_window = sinc((pi * x) / (scale * a))
    kernel1d = wider_sinc * lanczos_window

    # normalization step, so that we don't boost or attenuate the image
    kernel1d /= sum(kernel1d)
    kernel2d = (kernel1d[:, None] @ kernel1d[None, :])[None, None].repeat(
        n_channels, 1, 1, 1
     )

    return kernel2d


def pad_image(image: torch.Tensor, a: int = 2, scale: int = 2) -> torch.Tensor:
    padding = (a * scale)

    # add batch dimension if necessary
    image_batch = image if len(image.shape) == 4 else image[None]
    return F.pad(
        image_batch.permute(0, 3, 1, 2),
        [padding, padding - 1, padding, padding - 1],
        mode='reflect'
    )


def sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x == 0.0,
        torch.ones_like(x),
        x.sin() / x
    )


if __name__ == "__main__":
    main()
