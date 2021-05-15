import argparse
import os
import re
import sys
import datetime
from typing import Callable, Tuple

import numpy as np  # type: ignore
import h5py         # type: ignore

import utilities
from utilities import Metadata, HDF5Params
import constants


def main(args=None):
    if args is None:
        args = parse_arguments()

    if args.merge:
        # merging submatrices
        metadata = extract_metadata_submatrix(args)
        matrix_shape = (
            metadata.image_height, metadata.image_width, 3, metadata.n_bases
        )

        # make the hdf5 cache half the requested size
        # because 2 hdf5 files will be open at the same time
        hdf5_params = HDF5Params(
            matrix_shape, insert_dim=3, cache_size=args.cache_size / 2,
            rdcc_nslots=args.rdcc_nslots
        )

        merge_matrices(args, metadata, matrix_shape, hdf5_params)
    else:
        # building a matrix out of basis images
        metadata = extract_metadata(args)
        matrix_shape = (
            metadata.image_height, metadata.image_width, 3, metadata.n_bases
        )
        hdf5_params = HDF5Params(
            matrix_shape, insert_dim=3, cache_size=args.cache_size,
            rdcc_nslots=args.rdcc_nslots
        )

        build_matrix(args, metadata, hdf5_params)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dir',
        help='Directory containing basis image subdirectories.'
    )

    parser.add_argument(
        '--pattern',
        default='^([0-9]+)-([0-9]+)$',
        help=(
            'A regex pattern that matches the subdirectories inside --dir'
            'and extracts the "x" and "y" coordinates, respectively.'
        )
    )

    parser.add_argument(
        '--filename',
        default='scene.npy',
        help=(
            'Filename of the basis images. '
            'Extensions .exr and .npy are accepted.'
        )
    )

    parser.add_argument(
        '--output',
        help='Output directory. If empty, --dir will be used.'
    )

    parser.add_argument(
        '--basis_width',
        default=-1,
        type=int,
        help=(
            'Width of the basis texture in pixels. '
            '--basis_width * --basis_height = total number of basis images. '
            'Has to be set when --pattern is not supposed to '
            'match all basis subdirectories. '
            'Otherwise it is deduced automatically.'
        )
    )

    parser.add_argument(
        '--basis_height',
        type=int,
        default=-1,
        help=(
            'Height of the basis texture in pixels. '
            '--basis_width * --basis_height = total number of basis images. '
            'Has to be set when --pattern is not supposed to '
            'match all basis subdirectories. '
            'Otherwise it is deduced automatically.'
        )
    )

    parser.add_argument(
        '-q',
        dest='quiet',
        action='store_true',
        help=('Do not output anything to the console.')
    )
    parser.set_defaults(quiet=False)

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
        '--mode',
        default='a',
        choices=['a', 'w'],
        help=(
            'h5py file mode. "a" appends to a matrix file if it exists. '
            '"w" overwrites a matrix file if it exists. '
            'If the file does not exist, it will be created in any case.'
        )
    )

    parser.add_argument(
        '--merge',
        dest='merge',
        action='store_true',
        help='Merge from submatrices instead of building from basis images.'
    )
    parser.set_defaults(merge=False)

    args = parser.parse_args()
    if args.output is None:
        args.output = args.dir

    _, ext = os.path.splitext(args.filename)
    if ext != '.exr' and ext != '.npy':
        sys.exit('Invalid basis image file extension in --filename!')

    if ((args.basis_width != -1 and args.basis_height == -1) or
       (args.basis_width == -1 and args.basis_height != -1)):
        sys.exit('--basis_height needs to be set iff --basis_width is set!')

    return args


def build_matrix(
    args, metadata: Metadata, hdf5_params: HDF5Params,
    basis_image_loader:
        Callable[[dict], np.ndarray]
        = lambda basis_file: load_basis_image(basis_file)
):
    '''
    Loads basis images, stacks them into a light transport matrix
    and writes this matrix into an HDF5 file. Adds an image containing
    extra illumination if it is present in the input folder.
    '''
    with h5py.File(
            os.path.join(args.output, constants.MATRIX_FILENAME),
            mode=args.mode, rdcc_nbytes=hdf5_params.rdcc_nbytes,
            rdcc_w0=hdf5_params.rdcc_w0, rdcc_nslots=hdf5_params.rdcc_nslots
    ) as f:
        matrix = create_datasets(f, hdf5_params, metadata)

        # stack basis images into the matrix columns
        counter = 0
        next_percentage = 0.1
        total = len(metadata.files)
        for basis_file in metadata.files:
            image = basis_image_loader(basis_file)[:, :, :, np.newaxis]
            i = compute_insert_index(
                metadata.basis_height,
                basis_file['coord_x'],
                basis_file['coord_y']
            )

            # matrix[:, :, :, i] = image is much slower!
            matrix[:, :, :, i:i+1] = image

            # report progress if requested
            if not args.quiet:
                counter += 1
                if counter / total >= next_percentage:
                    print('{:.2f}% of the matrix is built ({})'.format(
                        next_percentage * 100, datetime.datetime.now()
                    ))
                    next_percentage += 0.1


def merge_matrices(
    args, metadata: Metadata, matrix_shape: Tuple[int, int, int, int],
    hdf5_params: HDF5Params
):
    '''
    Merges all submatrices from the input folder into a single matrix.
    Adds an image containing extra illumination if it is present
    in the input folder.
    '''
    matrix = np.zeros(matrix_shape, np.float32)

    with h5py.File(
            os.path.join(args.dir, constants.MATRIX_FILENAME), mode='w',
            rdcc_nbytes=hdf5_params.rdcc_nbytes, rdcc_w0=hdf5_params.rdcc_w0,
            rdcc_nslots=hdf5_params.rdcc_nslots
    ) as f_out:
        matrix = create_datasets(f_out, hdf5_params, metadata)

        total = len(metadata.files)
        counter = 1
        for submatrix_item in metadata.files:
            with h5py.File(
                submatrix_item['path'], mode='r',
                rdcc_nbytes=hdf5_params.rdcc_nbytes,
                rdcc_w0=hdf5_params.rdcc_w0,
                rdcc_nslots=hdf5_params.rdcc_nslots
            ) as f_in:
                i = submatrix_item['start']
                j = submatrix_item['end']

                submatrix = f_in[constants.MATRIX_DATASET_NAME]
                matrix[:, :, :, i:j] = submatrix[:, :, :, i:j]

                if not args.quiet:
                    print('{} out of {} matrices merged ({})'.format(
                        counter, total, datetime.datetime.now()
                    ))
                    counter += 1


def compute_insert_index(
    basis_height: int, basis_pos_x: int, basis_pos_y: int
) -> int:
    return basis_height * basis_pos_x + basis_pos_y


def extract_metadata(args) -> Metadata:
    # do a pass over args.dir to extract some useful info
    image_height = 0
    image_width = 0
    basis_height = args.basis_height
    basis_width = args.basis_width

    files = []
    extra_illum = None
    for item in os.listdir(args.dir):
        if re.match(args.pattern, item) is not None:
            coords = item.split('-')
            coord_x = (int(coords[0]) - 1)  # count from 0
            coord_y = (int(coords[1]) - 1)  # count from 0

            # extract paths to the basis image file and its coordinates
            basis_file = {
                'path': os.path.join(args.dir, item, args.filename),
                'coord_x': coord_x,
                'coord_y': coord_y
            }
            files.append(basis_file)

            # extract basis image dimensions
            if image_height == 0:
                basis_image = load_basis_image(basis_file)
                image_height, image_width, _ = basis_image.shape

            # extract other information from the folder contents
            basis_width = max(basis_width, coord_x + 1)
            basis_height = max(basis_height, coord_y + 1)
        elif re.match(constants.EXTRA_ILLUM_FILENAME, item):
            extra_illum = np.load(os.path.join(args.dir, item))

    return Metadata(
        files, basis_width, basis_height, image_width, image_height,
        extra_illum
    )


def extract_metadata_submatrix(args) -> Metadata:
    # assumptions:
    #   1) submatrices contain vertical slices of the whole basis
    #   2) submatrices in args.dir together always contain all basis images

    submatrix_filenames = []
    extra_illum = None
    for item in os.listdir(args.dir):
        if re.match(constants.SUBMATRIX_PATTERN, item):
            submatrix_filenames.append(item)
        elif re.match(constants.EXTRA_ILLUM_FILENAME, item):
            extra_illum = np.load(os.path.join(args.dir, item))
    submatrix_filenames.sort()

    files = []
    image_width = -1
    image_height = -1
    basis_width = 0
    basis_height = -1

    for submatrix_filename in submatrix_filenames:
        submatrix_path = os.path.join(args.dir, submatrix_filename)
        with h5py.File(submatrix_path, mode='r') as f:
            if basis_height == -1:
                basis_height = f[constants.METADATA_DATASET_NAME][1]
            files.append(
                {
                    'path': submatrix_path,
                    'start': basis_height * basis_width,
                    'end': basis_height * f[constants.METADATA_DATASET_NAME][0]
                }
            )
            if image_width == -1:
                image_width = f[constants.METADATA_DATASET_NAME][2]
            if image_height == -1:
                image_height = f[constants.METADATA_DATASET_NAME][3]
            basis_width = f[constants.METADATA_DATASET_NAME][0]

    return Metadata(
        files, basis_width, basis_height, image_width, image_height,
        extra_illum
    )


def create_datasets(
    matrix_file: h5py.File, hdf5_params: HDF5Params, metadata: Metadata
) -> np.ndarray:
    # create metadata dataset if it doesn't exist
    if constants.METADATA_DATASET_NAME not in matrix_file:
        metadata_array = metadata.get_numpy_array()
        matrix_file.create_dataset(
            constants.METADATA_DATASET_NAME,
            shape=metadata_array.shape,
            dtype=np.uint32,
            data=metadata_array
        )

    # create matrix dataset if it doesn't exist
    if constants.MATRIX_DATASET_NAME not in matrix_file:
        matrix_file.create_dataset(
            constants.MATRIX_DATASET_NAME,
            shape=hdf5_params.matrix_shape,
            chunks=tuple(hdf5_params.chunk_shape),
            dtype=np.float32
        )

    # create extra illumination dataset if it doesn't exist
    if (
        constants.EXTRA_ILLUM_DATASET_NAME not in matrix_file and
        metadata.extra_illum is not None
    ):
        matrix_file.create_dataset(
            constants.EXTRA_ILLUM_DATASET_NAME,
            data=metadata.extra_illum
        )

    return matrix_file[constants.MATRIX_DATASET_NAME]


def load_basis_image(basis_file: dict) -> np.ndarray:
    file_path = basis_file['path']
    _, ext = os.path.splitext(file_path)
    if ext == '.exr':
        return utilities.read_image(file_path)
    elif ext == '.npy':
        return np.load(file_path)


if __name__ == "__main__":
    main()
