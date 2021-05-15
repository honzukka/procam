import argparse
import os
import time

import numpy as np      # type: ignore
import torch

import utilities
from utilities import Metadata, HDF5Params
import lanczos


def main(args=None):
    if args is None:
        args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metadata = Metadata.create_from_matrix_file(args.dir)

    # load content
    content = None
    if not args.video:
        content = load_image(args, metadata)
    else:
        content = load_video(args, metadata)

    # decide if matrix loading and content projection
    # should be done at once or in parts (based on cache size)
    result = None
    hdf5_params = HDF5Params.create_from_metadata(
        metadata, args.cache_size, args.rdcc_nslots
    )
    if hdf5_params.n_chunks_in_cache < hdf5_params.n_chunks:
        start = time.time()
        h, w, _, _ = hdf5_params.matrix_shape
        n = content.shape[0]
        result = torch.zeros((n, h, w, 3), dtype=torch.float32, device=device)
        i = 0
        while i < hdf5_params.n_chunks:
            j = i + hdf5_params.n_chunks_in_cache
            matrix_chunk = utilities.load_matrix_torch(
                args.dir, hdf5_params, i, j
            ).to(device)
            content_chunk = content[:, i:j, :].to(device)

            result += project(content_chunk, matrix_chunk, args.brightness)
            i = j
            if not args.quiet:
                print('{} out of {} chunks done ({:.2f}s)'.format(
                    min(i, hdf5_params.n_chunks), hdf5_params.n_chunks,
                    time.time() - start
                ))
    else:
        start = time.time()
        matrix = utilities.load_matrix_torch(args.dir, hdf5_params).to(device)
        content = content.to(device)
        if not args.quiet:
            print('Matrix loaded in {}s'.format(time.time() - start))

        start = time.time()
        result = project(content, matrix, args.brightness)
        if not args.quiet:
            print('Multiplication done in {}s'.format(time.time() - start))

    # numpy doesn't like it when Tensors are on the GPU and
    # OpenEXR doesn't like when arrays are not contiguous,
    # so let's not make them mad
    result_numpy = np.ascontiguousarray(
        result
        .cpu()
        .numpy()
        .squeeze()  # get rid of dim "N" in case the content is only 1 image
    )
    if not args.video:
        utilities.write_image(
            os.path.join(args.out_dir, 'output.exr'),
            result_numpy
        )
    else:
        utilities.write_video(
            os.path.join(args.out_dir, 'output.mp4'),
            result_numpy
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dir',
        help=(
            'Directory containing the light transport matrix. '
            'This is also the output directory unless --out_dir is specified.'
        )
    )

    parser.add_argument(
        '--out_dir',
        default='',
        help='Output directory. If empty, dir will be used.'
    )

    parser.add_argument(
        '--in_name',
        required=True,
        help=(
            'Name of the image/video inside --dir to be projected. '
            'Can also be an .npy file.'
        )
    )

    parser.add_argument(
        '--brightness',
        default=1.0,
        help='Projector brightness. It is used to multiply the result.'
    )

    parser.add_argument(
        '--cache_size',
        type=int,
        default=2000,
        help=(
            'HDF5 cache size in MB. '
            'Should be as large as possible but within available RAM.'
        )
    )

    parser.add_argument(
        '--rdcc_nslots',
        type=int,
        default=-1,
        help=(
            'This is an h5py parameter '
            '(http://docs.h5py.org/en/stable/high/file.html#chunk-cache). '
            'This script sets it automatically to be the correct size, '
            'but not a prime number as that is too time-consuming. '
            'If you know what you are doing, you can set it manually '
            'for optimal caching performance via this argument.'
        )
    )

    parser.add_argument(
        '--video',
        dest='video',
        action='store_true',
        help='Set if --in_name refers to a video.'
    )
    parser.set_defaults(video=False)

    parser.add_argument(
        '-q',
        dest='quiet',
        action='store_true',
        help=('Do not output anything to the console.')
    )
    parser.set_defaults(quiet=False)

    args = parser.parse_args()
    if args.out_dir == '':
        args.out_dir = args.dir

    return args


def project(
    content: torch.Tensor, matrix: torch.Tensor, brightness: float = 1.0
) -> torch.Tensor:
    # content needs to have shape [N, H * W, C]
    if len(content.shape) == 2:
        content = content[None, :, :]

    assert len(content.shape) == 3

    # note: this is not the most optimal way to perform matrix multiplication,
    # but an equivalent matmul is only negligibly faster, so we opt for
    # convenience here
    return brightness * torch.einsum(
        'hwcb,nbc->nhwc',
        matrix,
        content
    )


def load_image(args, metadata: Metadata) -> torch.Tensor:
    _, ext = os.path.splitext(args.in_name)
    file_path = os.path.join(args.dir, args.in_name)
    image = None
    if ext == '.npy':
        image = np.load(file_path, dtype=np.float32)
    else:
        image = utilities.read_image(file_path)

    image = torch.from_numpy(image)

    image = lanczos.scale_to_target(
        image, (metadata.basis_height, metadata.basis_width)
    )

    return image.permute(0, 2, 1, 3).reshape((1, -1, 3))


def load_video(args, metadata: Metadata) -> torch.Tensor:
    filename = os.path.join(args.dir, args.in_name)
    video = utilities.read_video(filename, metadata.basis_width)
    video = torch.from_numpy(video)
    video = lanczos.scale_to_target(
        video, (metadata.basis_height, metadata.basis_width)
    )

    n_frames = video.shape[0]
    return video.permute(0, 2, 1, 3).reshape((n_frames, -1, 3))


if __name__ == "__main__":
    main()
