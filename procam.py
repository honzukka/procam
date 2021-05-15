import argparse
from typing import Optional
import os

import torch

import models
from optimizers import OptimizerSciPy
import constants
from log import Logger
from loader import Loader


def main(config: Optional[argparse.Namespace] = None):
    if config is None:
        config = parse_arguments()
    config = process_arguments(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.mode == 'comparison_proj':
        # pure compensation
        config.mode = 'comp_proj'
        config.suffix = config.mode
        procam(config, device)

        # projection synthesis
        config.mode = 'syn_proj'
        config.suffix = config.mode
        procam(config, device)
    elif config.mode == 'comparison_tex':
        # pure compensation
        config.mode = 'comp_tex'
        config.suffix = config.mode
        procam(config, device)

        # texture synthesis
        config.mode = 'syn_tex'
        config.suffix = config.mode
        procam(config, device)
    else:
        procam(config, device)


def procam(config: argparse.Namespace, device: torch.device):
    loader = Loader(config)

    for i in range(config.n_scenes):
        config.scene_ind = i
        print('Scene set to: {}'.format(config.scene_names[i]))

        # load model & data
        data = loader.load(i)
        if config.mode == 'syn_tex':
            model = models.TextureSynthesisModel(data, config, device)
        elif config.mode == 'syn_proj':
            model = models.ProjectionSynthesisModel(data, config, device)
        elif config.mode == 'comp_tex':
            model = models.TextureCompensationModel(data, config, device)
        elif config.mode == 'comp_proj':
            model = models.ProjectionCompensationModel(data, config, device)
        else:
            raise ValueError('Invalid mode!')

        for j in model:
            config.target_ind = j
            print('Target set to: {}'.format(config.target_names[j]))

            logger = Logger(data, config)
            optimizer = OptimizerSciPy(model, logger, config)
            results = optimizer.optimize()
            logger.save(*results)

        # free up memory for the next scene
        del data, model, optimizer, logger
        torch.cuda.empty_cache()


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'mode',
        choices=[
            'syn_tex', 'syn_proj', 'comp_tex', 'comp_proj',
            'comparison_proj', 'comparison_tex'
        ],
        help=(
            'The model to be optimized. '
            '[syn_tex] synthesizes a texture that matches a target '
            'with an optional projection onto a background. '
            '[syn_proj] synthesizes a texture that matches a target '
            'when projected onto a scene defined by an LT matrix. '
            '[comp_tex] returns a texture that matches a target '
            'when projected onto a background. '
            '[comp_proj] returns a texture that matches a target '
            'when projected onto a scene defined by an LT matrix. '
            '[comparison_proj] syn_proj & comp_proj '
            '[comparison_tex] syn_tex & comp_tex '
        )
    )

    # INPUT & OUTPUT
    # ------------------------------------------------------

    parser.add_argument(
        '-t',
        dest='target_paths',
        required=True,
        nargs='*',
        help='Paths to the target images.'
    )

    parser.add_argument(
        '-m',
        dest='model_path',
        default=None,
        help='Path to the VGG model used for texture synthesis.'
    )

    parser.add_argument(
        '-l',
        dest='lt_matrix_paths',
        default=None,
        nargs='*',
        help='Paths to light transport (LT) matrices.'
    )

    parser.add_argument(
        '-b',
        dest='background_paths',
        default=None,
        nargs='*',
        help='Paths to the background images.'
    )

    parser.add_argument(
        '-o',
        dest='out_dir',
        default='output',
        help='Output directory.'
    )

    # MODEL SETTINGS
    # ------------------------------------------------------

    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        help=(
            'Number of images produced (each coming from a different seed). '
            'Applies only to [syn_proj] and [syn_tex].'
        )
    )

    parser.add_argument(
        '--pyramid_w',
        dest='pyramid_weights',
        type=float,
        default=[1.0],
        nargs='*',
        help=(
            'The weight of each pyramid layer in the loss. '
            'Applies only to [syn_proj] and [syn_tex].'
        )
    )

    parser.add_argument(
        '--layer',
        dest='loss_layers',
        nargs='*',
        default=['0', '1', '2', '3', '4'],
        help=(
            'Indices of layers to be used in the loss. '
            '"1:2" means that layer pool1 will be chained with layer pool2. '
            'Applies only to [syn_proj] and [syn_tex]. '
            'Layer list: {}'.format(constants.LOSS_LAYERS)
        )
    )

    parser.add_argument(
        '--layer_w',
        dest='layer_weights',
        nargs='*',
        type=float,
        default=1e08,
        help=(
            'The weight of each network layer in the loss. '
            'If only one value is supplied, it is applied uniformly. '
            'Otherwise it is applied per layer. '
            'Applies only to [syn_proj] and [syn_tex].'
        )
    )

    parser.add_argument(
        '--offset',
        dest='gram_offset',
        type=float,
        default=-1.0,
        help=(
            'This number is added to activations before Gram matrix '
            'computation. Applies only to [syn_proj] and [syn_tex].'
        )
    )

    parser.add_argument(
        '--bright',
        dest='base_brightness',
        type=float,
        default=1.0,
        help='Basic multiplier for all rendering.'
    )

    parser.add_argument(
        '--size',
        dest='opt_size',
        type=int,
        nargs=2,
        metavar=('x', 'y'),
        default=None,
        help=(
            'Resolution of the optimized output. Applies only to [syn_tex]. '
            '([width, height])'
        )
    )

    parser.add_argument(
        '--cache',
        dest='cache_size',
        type=int,
        default=1000,
        help=(
            'HDF5 cache size in MB. '
            'Should be as large as possible but within available RAM.'
        )
    )

    parser.add_argument(
        '--gpus',
        dest='n_gpus',
        type=int,
        default=1,
        help=(
            'When working with projections, this argument can be used '
            'to split the light transport matrix across multiple gpus.'
        )
    )

    # OPTIMIZER SETTINGS
    # ------------------------------------------------------

    parser.add_argument(
        '--check',
        dest='checkpoint_every',
        type=int,
        default=1,
        help=(
            'The number of iterations between checkpoints '
            'where progress information and intermediate values are stored.'
        )
    )

    parser.add_argument(
        '--full_out',
        dest='full_output',
        type=bool,
        default=True,
        help=(
            'If set to True, a potentially very large HDF5 file '
            'with all intermediate results will be created.'
        )
    )

    parser.add_argument(
        '--step',
        dest='n_steps',
        type=int,
        default=100,
        help='The maximum number of optimizer steps to be performed.'
    )

    config = parser.parse_args()

    # create output subfolder
    config.out_dir = os.path.join(
        config.out_dir, '{}'.format(config.mode)
    )
    os.makedirs(config.out_dir, exist_ok=True)

    return config


def process_arguments(config: argparse.Namespace) -> argparse.Namespace:
    if (
        config.lt_matrix_paths is not None and
        config.background_paths is not None
    ):
        raise RuntimeError(
            'LT matrix paths and background paths cannot '
            'both be set at the same time!'
        )

    config.n_scenes = 1
    config.scene_names = ['blank_scene']

    # extract the amount of scenes and their names
    if config.lt_matrix_paths is not None:
        config.n_scenes = len(config.lt_matrix_paths)
        config.scene_names = [
            os.path.split(os.path.split(scene_path)[0])[1]
            for scene_path in config.lt_matrix_paths
        ]
    if config.background_paths is not None:
        config.n_scenes = len(config.background_paths)
        config.scene_names = [
            os.path.splitext(os.path.basename(scene_path))[0]
            for scene_path in config.background_paths
        ]

    # extract target names
    config.target_names = [
        os.path.splitext(os.path.basename(target_path))[0]
        for target_path in config.target_paths
    ]

    # make sure pyramid weights are in a list and deduce the number of
    # pyramid layers from the size of the list
    assert isinstance(config.pyramid_weights, list)
    config.pyramid_layers = len(config.pyramid_weights)

    # make sure there is a weight per each loss layer/pair of layers
    if isinstance(config.layer_weights, float):
        config.layer_weights = [
            config.layer_weights for i in range(len(config.loss_layers))
        ]
    assert len(config.layer_weights) == len(config.loss_layers)

    # convert loss layer index strings to int pairs
    int_pairs = []
    for layer_ind_str in config.loss_layers:
        layer_pair_str = layer_ind_str.split(':')
        assert len(layer_pair_str) <= 2     # only "L1" and "L1:L2" allowed
        layer_pair_int = tuple([int(s) for s in layer_pair_str])
        if len(layer_pair_int) == 2:
            int_pairs.append(layer_pair_int)
        else:   # convert (k) to (k, k)
            int_pairs.append((layer_pair_int[0], layer_pair_int[0]))

    # extract unique layer names
    config.layer_names = []
    for pair in int_pairs:
        config.layer_names += [constants.LOSS_LAYERS[i] for i in pair]
    config.layer_names = list(dict.fromkeys(config.layer_names))  # unique

    # convert layer index int pairs, so that they index into
    # the subset of chosen layers
    unique_indices = []
    for ind1, ind2 in int_pairs:
        if ind1 not in unique_indices:
            unique_indices.append(ind1)
        if ind2 not in unique_indices:
            unique_indices.append(ind2)
    unique_indices.sort()
    converted_int_pairs = []
    for ind1, ind2 in int_pairs:
        converted_int_pairs.append(
            (unique_indices.index(ind1), unique_indices.index(ind2))
        )
    config.loss_layers = converted_int_pairs

    # load crop information for each scene
    config.crops = get_crops_from_files(config.lt_matrix_paths)

    return config


# the user is expected to create a file (constants.CROP_FILENAME)
# if they want images to be cropped after rendering and before loss computation
def get_crops_from_files(matrix_paths):
    # no crop if mode is [syn_tex] or [comp_tex]
    if not isinstance(matrix_paths, list):
        return None

    folders = [os.path.dirname(matrix_path) for matrix_path in matrix_paths]
    crops = []
    for folder in folders:
        crop_filename = os.path.join(folder, constants.CROP_FILENAME)
        if os.path.isfile(crop_filename):
            with open(crop_filename) as f:
                crop_str = f.read().strip().split(' ')
                crop = [int(x) for x in crop_str]
            assert len(crop) == 4
            crops.append(crop)
            print('Crop {}: {}'.format(folder, crop))
        else:
            crops.append(None)
            print('Crop {} not found. No crop will be set.'.format(folder))
    return crops


if __name__ == "__main__":
    main()
