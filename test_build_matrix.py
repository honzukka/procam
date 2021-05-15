import shutil

import numpy as np
import h5py
import pytest

import build_matrix
import constants


class DummyArgs:
    def __init__(
        self, directory, filename, pattern, basis_height=-1, basis_width=-1,
        dummy=[-1 for i in range(8)], output=None, mode='a'
    ):
        self.dir = directory
        self.output = output if output is not None else directory
        self.filename = filename
        self.pattern = pattern
        self.basis_height = basis_height
        self.basis_width = basis_width
        self.dummy = dummy
        self.quiet = True
        self.cache_size = 500
        self.mode = mode


class DummyMetadata:
    def __init__(
        self, files, basis_width, basis_height, image_width, image_height,
        n_bases, n_pixels
    ):
        self.files = files
        self.basis_width = basis_width
        self.basis_height = basis_height
        self.image_width = image_width
        self.image_height = image_height
        self.n_bases = n_bases
        self.n_pixels = n_pixels
        self.extra_illum = None

    def get_numpy_array(self):
        return np.array(
            [
                self.basis_width,
                self.basis_height,
                self.image_width,
                self.image_height
            ]
        )


class DummyHDF5Params:
    def __init__(
        self, rdcc_nbytes, rdcc_w0, rdcc_nslots,
        matrix_shape, chunk_shape
    ):
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_w0 = rdcc_w0
        self.rdcc_nslots = rdcc_nslots
        self.matrix_shape = matrix_shape
        self.chunk_shape = chunk_shape


def create_dummy_files(
    tmp_path, image_shape, basis_range, seed=1234, basis_start=(0, 0)
):
    basis_end = (
        basis_start[0] + basis_range[0], basis_start[1] + basis_range[1]
    )
    for i in range(basis_start[0], basis_end[0]):
        for j in range(basis_start[1], basis_end[1]):
            subdir = tmp_path.joinpath('{:04d}-{:04d}'.format(i + 1, j + 1))
            if not subdir.exists():
                subdir.mkdir()
            image_path = subdir.joinpath('scene.npy')
            np.save(
                image_path,
                np.random.default_rng(seed).random(
                    image_shape, dtype=np.float32
                )
            )


class TestExtractMetadata:
    def is_metadata_equal_numbers(self, metadata_actual, metadata_expected):
        return (
            metadata_actual.basis_width == metadata_expected[0] and
            metadata_actual.basis_height == metadata_expected[1] and
            metadata_actual.image_width == metadata_expected[2] and
            metadata_actual.image_height == metadata_expected[3] and
            metadata_actual.n_bases == metadata_expected[4] and
            metadata_actual.n_pixels == metadata_expected[5]
        )

    def is_metadata_equal_files(self, metadata_actual, metadata_expected):
        is_equal = True
        extract_key = lambda item: (item['coord_x'], item['coord_y'])
        actual_sorted = sorted(metadata_actual.files, key=extract_key)
        expected_sorted = sorted(metadata_expected, key=extract_key)
        # for file1, file2 in zip(metadata_actual.files, metadata_expected):
        for file1, file2 in zip(actual_sorted, expected_sorted):
            for key in file1:
                is_equal &= (file1[key] == file2[key])
        return is_equal

    def test_numbers_square(self, tmp_path):
        basis_size = 3
        image_size = 640
        args = DummyArgs(tmp_path, 'scene.npy', '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        ]

        assert self.is_metadata_equal_numbers(actual, expected), \
            'actual: {}, expected: {}'.format(actual, expected)

    def test_numbers_rectangle_basis(self, tmp_path):
        basis_width = 3
        basis_height = 2
        image_size = 640
        args = DummyArgs(tmp_path, 'scene.npy', '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_width, basis_height)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            basis_width, basis_height, image_size, image_size,
            basis_width * basis_height, image_size**2
        ]

        assert self.is_metadata_equal_numbers(actual, expected), \
            'actual: {}, expected: {}'.format(actual, expected)

    def test_numbers_rectangle_image(self, tmp_path):
        basis_size = 3
        image_width = 640
        image_height = 480
        args = DummyArgs(tmp_path, 'scene.npy', '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_height, image_width, 3),
            (basis_size, basis_size)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            basis_size, basis_size, image_width, image_height,
            basis_size**2, image_width * image_height
        ]

        assert self.is_metadata_equal_numbers(actual, expected), \
            'actual: {}, expected: {}'.format(actual, expected)

    def test_files_square(self, tmp_path):
        basis_size = 2
        image_size = 640
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            {
                'path': str(tmp_path.joinpath('0001-0001', filename)),
                'coord_x': 0, 'coord_y': 0
            },
            {
                'path': str(tmp_path.joinpath('0001-0002', filename)),
                'coord_x': 0, 'coord_y': 1
            },
            {
                'path': str(tmp_path.joinpath('0002-0001', filename)),
                'coord_x': 1, 'coord_y': 0
            },
            {
                'path': str(tmp_path.joinpath('0002-0002', filename)),
                'coord_x': 1, 'coord_y': 1
            }
        ]

        assert self.is_metadata_equal_files(actual, expected), \
            'actual: {}, expected: {}'.format(actual.files, expected)

    def test_files_rectangle_basis(self, tmp_path):
        basis_width = 2
        basis_height = 1
        image_size = 640
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_width, basis_height)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            {
                'path': str(tmp_path.joinpath('0001-0001', filename)),
                'coord_x': 0, 'coord_y': 0
            },
            {
                'path': str(tmp_path.joinpath('0002-0001', filename)),
                'coord_x': 1, 'coord_y': 0
            },
        ]

        assert self.is_metadata_equal_files(actual, expected), \
            'actual: {}, expected: {}'.format(actual.files, expected)

    def test_files_rectangle_image(self, tmp_path):
        basis_size = 2
        image_width = 640
        image_height = 480
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_width, image_height, 3),
            (basis_size, basis_size)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            {
                'path': str(tmp_path.joinpath('0001-0001', filename)),
                'coord_x': 0, 'coord_y': 0
            },
            {
                'path': str(tmp_path.joinpath('0001-0002', filename)),
                'coord_x': 0, 'coord_y': 1
            },
            {
                'path': str(tmp_path.joinpath('0002-0001', filename)),
                'coord_x': 1, 'coord_y': 0
            },
            {
                'path': str(tmp_path.joinpath('0002-0002', filename)),
                'coord_x': 1, 'coord_y': 1
            }
        ]

        assert self.is_metadata_equal_files(actual, expected), \
            'actual: {}, expected: {}'.format(actual.files, expected)

    def test_numbers_partial_build(self, tmp_path):
        basis_size = 3
        image_size = 640
        filename = 'scene.npy'
        args = DummyArgs(
            tmp_path, filename, '^0001-0001$', basis_size, basis_size
        )
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        ]

        assert self.is_metadata_equal_numbers(actual, expected), \
            'actual: {}, expected: {}'.format(actual, expected)

    def test_files_partial_build(self, tmp_path):
        basis_size = 3
        image_size = 640
        filename = 'scene.npy'
        args = DummyArgs(
            tmp_path, filename, '^0001-0001$', basis_size, basis_size
        )
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            {
                'path': str(tmp_path.joinpath('0001-0001', filename)),
                'coord_x': 0, 'coord_y': 0
            }
        ]

        assert self.is_metadata_equal_files(actual, expected), \
            'actual: {}, expected: {}'.format(actual.files, expected)

    def test_numbers_partial_build_no_basis_size(self, tmp_path):
        basis_size = 1
        image_size = 640
        filename = 'scene.npy'
        args = DummyArgs(
            tmp_path, filename, '^0001-0001$'
        )
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        ]

        assert self.is_metadata_equal_numbers(actual, expected), \
            'actual: {}, expected: {}'.format(actual, expected)

    def test_numbers_partial_build_no_basis_size_no_beginning(self, tmp_path):
        basis_size = 2
        image_size = 640
        filename = 'scene.npy'
        args = DummyArgs(
            tmp_path, filename, '^0002-0002$'
        )
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )

        actual = build_matrix.extract_metadata(args)
        expected = [
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        ]

        assert self.is_metadata_equal_numbers(actual, expected), \
            'actual: {}, expected: {}'.format(actual, expected)


class TestComputeInsertIndex:
    def test_index_unit(self):
        basis_height = 1
        basis_pos_x = 0
        basis_pos_y = 0

        actual = build_matrix.compute_insert_index(
            basis_height, basis_pos_x, basis_pos_y
        )
        expected = 0

        assert actual == expected

    def test_index_end(self):
        basis_height = 4
        basis_pos_x = 5
        basis_pos_y = 3

        actual = build_matrix.compute_insert_index(
            basis_height, basis_pos_x, basis_pos_y
        )
        expected = 23

        assert actual == expected

    def test_index_column_end(self):
        basis_height = 4
        basis_pos_x = 0
        basis_pos_y = 3

        actual = build_matrix.compute_insert_index(
            basis_height, basis_pos_x, basis_pos_y
        )
        expected = 3

        assert actual == expected

    def test_index_row_end(self):
        basis_height = 4
        basis_pos_x = 5
        basis_pos_y = 0

        actual = build_matrix.compute_insert_index(
            basis_height, basis_pos_x, basis_pos_y
        )
        expected = 20

        assert actual == expected

    def test_index(self):
        basis_height = 4
        basis_pos_x = 4
        basis_pos_y = 2

        actual = build_matrix.compute_insert_index(
            basis_height, basis_pos_x, basis_pos_y
        )
        expected = 18

        assert actual == expected


@pytest.mark.slow
class TestBuildMatrix:
    def test_square_basis_full(self, tmp_path):
        basis_size = 2
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )
        metadata = DummyMetadata(
            [
                {
                    'path': str(tmp_path.joinpath('0001-0001', filename)),
                    'coord_x': 0, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0001-0002', filename)),
                    'coord_x': 0, 'coord_y': 1
                },
                {
                    'path': str(tmp_path.joinpath('0002-0001', filename)),
                    'coord_x': 1, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0002-0002', filename)),
                    'coord_x': 1, 'coord_y': 1
                }
            ],
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            500 * 1024 * 1024, 1, 10007, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(args, metadata, hdf5_params)
        with h5py.File(tmp_path.joinpath(constants.MATRIX_FILENAME), 'r') as f:
            actual = f[constants.MATRIX_DATASET_NAME][()]
            expected = np.zeros(matrix_shape, dtype=np.float32)
            expected[:, :, :, :] = np.random.default_rng(1234).random(
                chunk_shape, dtype=np.float32
            )

            assert np.array_equal(actual, expected)

    def test_rectangle_basis_full(self, tmp_path):
        basis_width = 3
        basis_height = 1
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_width * basis_height)
        chunk_shape = (image_size, image_size, 3, 1)
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_width, basis_height)
        )
        metadata = DummyMetadata(
            [
                {
                    'path': str(tmp_path.joinpath('0001-0001', filename)),
                    'coord_x': 0, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0002-0001', filename)),
                    'coord_x': 1, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0003-0001', filename)),
                    'coord_x': 2, 'coord_y': 0
                }
            ],
            basis_width, basis_height, image_size, image_size,
            basis_width * basis_height, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            500 * 1024 * 1024, 1, 10007, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(args, metadata, hdf5_params)
        with h5py.File(tmp_path.joinpath(constants.MATRIX_FILENAME), 'r') as f:
            actual = f[constants.MATRIX_DATASET_NAME][()]
            expected = np.zeros(matrix_shape, dtype=np.float32)
            expected[:, :, :, :] = np.random.default_rng(1234).random(
                chunk_shape, dtype=np.float32
            )

            assert np.array_equal(actual, expected)

    def test_square_basis_partial(self, tmp_path):
        basis_size = 2
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size - 1)
        )
        metadata = DummyMetadata(
            [
                {
                    'path': str(tmp_path.joinpath('0001-0001', filename)),
                    'coord_x': 0, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0002-0001', filename)),
                    'coord_x': 1, 'coord_y': 0
                }
            ],
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            500 * 1024 * 1024, 1, 10007, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(args, metadata, hdf5_params)
        with h5py.File(tmp_path.joinpath(constants.MATRIX_FILENAME), 'r') as f:
            actual = f[constants.MATRIX_DATASET_NAME][()]
            expected = np.zeros(matrix_shape, dtype=np.float32)
            expected[:, :, :, 0:1] = np.random.default_rng(1234).random(
                chunk_shape, dtype=np.float32
            )
            expected[:, :, :, 2:3] = np.random.default_rng(1234).random(
                chunk_shape, dtype=np.float32
            )

            assert np.array_equal(actual, expected)

    def test_square_basis_full_write_mode(self, tmp_path):
        basis_size = 2
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$', mode='w')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )

        # build matrix with bases [0, 0] and [1, 0]
        metadata = DummyMetadata(
            [
                {
                    'path': str(tmp_path.joinpath('0001-0001', filename)),
                    'coord_x': 0, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0002-0001', filename)),
                    'coord_x': 1, 'coord_y': 0
                }
            ],
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            500 * 1024 * 1024, 1, 10007, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(args, metadata, hdf5_params)

        # build matrix with bases [0, 0] and [0, 1]
        metadata = DummyMetadata(
            [
                {
                    'path': str(tmp_path.joinpath('0001-0001', filename)),
                    'coord_x': 0, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0001-0002', filename)),
                    'coord_x': 0, 'coord_y': 1
                }
            ],
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            500 * 1024 * 1024, 1, 10007, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(args, metadata, hdf5_params)

        # test that matrix was indeed overwritten
        with h5py.File(tmp_path.joinpath(constants.MATRIX_FILENAME), 'r') as f:
            actual = f[constants.MATRIX_DATASET_NAME][()]
            expected = np.zeros(matrix_shape, dtype=np.float32)
            expected[:, :, :, 0:2] = np.random.default_rng(1234).random(
                chunk_shape, dtype=np.float32
            )

            assert np.array_equal(actual, expected)

    def test_square_basis_full_append_mode(self, tmp_path):
        basis_size = 2
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$', mode='a')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )

        # build matrix with basis [0, 0]
        metadata = DummyMetadata(
            [
                {
                    'path': str(tmp_path.joinpath('0001-0001', filename)),
                    'coord_x': 0, 'coord_y': 0
                }
            ],
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            500 * 1024 * 1024, 1, 10007, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(args, metadata, hdf5_params)

        # build matrix with basis [0, 1]
        metadata = DummyMetadata(
            [
                {
                    'path': str(tmp_path.joinpath('0001-0002', filename)),
                    'coord_x': 0, 'coord_y': 1
                }
            ],
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            500 * 1024 * 1024, 1, 10007, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(args, metadata, hdf5_params)

        # test that both bases are in the matrix
        with h5py.File(tmp_path.joinpath(constants.MATRIX_FILENAME), 'r') as f:
            actual = f[constants.MATRIX_DATASET_NAME][()]
            expected = np.zeros(matrix_shape, dtype=np.float32)
            expected[:, :, :, 0:2] = np.random.default_rng(1234).random(
                chunk_shape, dtype=np.float32
            )

            assert np.array_equal(actual, expected)

    def test_square_basis_full_metadata(self, tmp_path):
        basis_size = 2
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )
        metadata = DummyMetadata(
            [
                {
                    'path': str(tmp_path.joinpath('0001-0001', filename)),
                    'coord_x': 0, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0001-0002', filename)),
                    'coord_x': 0, 'coord_y': 1
                },
                {
                    'path': str(tmp_path.joinpath('0002-0001', filename)),
                    'coord_x': 1, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0002-0002', filename)),
                    'coord_x': 1, 'coord_y': 1
                }
            ],
            basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            500 * 1024 * 1024, 1, 10007, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(args, metadata, hdf5_params)
        with h5py.File(tmp_path.joinpath(constants.MATRIX_FILENAME), 'r') as f:
            actual = f[constants.METADATA_DATASET_NAME][()]
            expected = np.array([2, 2, 640, 640], dtype=np.uint32)

            assert np.array_equal(actual, expected)

    def test_rectangle_basis_full_metadata(self, tmp_path):
        basis_width = 3
        basis_height = 1
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_width * basis_height)
        chunk_shape = (image_size, image_size, 3, 1)
        filename = 'scene.npy'
        args = DummyArgs(tmp_path, filename, '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            tmp_path,
            (image_size, image_size, 3),
            (basis_width, basis_height)
        )
        metadata = DummyMetadata(
            [
                {
                    'path': str(tmp_path.joinpath('0001-0001', filename)),
                    'coord_x': 0, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0002-0001', filename)),
                    'coord_x': 1, 'coord_y': 0
                },
                {
                    'path': str(tmp_path.joinpath('0003-0001', filename)),
                    'coord_x': 2, 'coord_y': 0
                }
            ],
            basis_width, basis_height, image_size, image_size,
            basis_width * basis_height, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            500 * 1024 * 1024, 1, 10007, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(args, metadata, hdf5_params)
        with h5py.File(tmp_path.joinpath(constants.MATRIX_FILENAME), 'r') as f:
            actual = f[constants.METADATA_DATASET_NAME][()]
            expected = np.array([3, 1, 640, 640], dtype=np.uint32)

            assert np.array_equal(actual, expected)


@pytest.mark.benchmark
class TestBenchmarkBuildMatrix:
    # correctness test for building matrices
    # from dummy images generated on the fly
    def test_dummy(self, tmp_path):
        basis_size = 2
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        args = DummyArgs(tmp_path, '', '')
        files = [
            {'coord_x': x, 'coord_y': y}
            for x in range(basis_size)
            for y in range(basis_size)
        ]

        metadata = DummyMetadata(
            files, basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            2000 * 1024 * 1024, 1, 40009, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(
            args, metadata, hdf5_params,
            lambda basis_file: np.random.default_rng(1234).random(
                (image_size, image_size, 3), dtype=np.float32
            )
        )

        with h5py.File(tmp_path.joinpath(constants.MATRIX_FILENAME), 'r') as f:
            actual = f[constants.MATRIX_DATASET_NAME][()]
            expected = np.zeros(matrix_shape, dtype=np.float32)
            expected[:, :, :, :] = np.random.default_rng(1234).random(
                chunk_shape, dtype=np.float32
            )

            assert np.array_equal(actual, expected)

    def build_matrix_dummy_basis10(self, tmp_path):
        basis_size = 10
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        subdir = tmp_path.joinpath('dummy_basis10')
        subdir.mkdir()
        args = DummyArgs(subdir, '', '')
        files = [
            {'coord_x': x, 'coord_y': y}
            for x in range(basis_size)
            for y in range(basis_size)
        ]

        metadata = DummyMetadata(
            files, basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            2000 * 1024 * 1024, 1, 40009, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(
            args, metadata, hdf5_params,
            lambda basis_file: np.random.default_rng(1234).random(
                (image_size, image_size, 3), dtype=np.float32
            )
        )

        # clean up, so that the function can be run multiple times in a row
        shutil.rmtree(subdir)

    def test_dummy_basis10(self, benchmark, tmp_path):
        benchmark(self.build_matrix_dummy_basis10, tmp_path)
        assert True

    def build_matrix_dummy_basis100_random_subset100_small_cache(
        self, tmp_path
    ):
        basis_size = 100
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        subdir = tmp_path.joinpath('dummy_basis10')
        subdir.mkdir()
        args = DummyArgs(subdir, '', '')
        files = [
            {'coord_x': x, 'coord_y': y}
            for x in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
            for y in [51, 53, 55, 57, 59, 61, 63, 65, 67, 69]
        ]

        assert len(files) == 100

        metadata = DummyMetadata(
            files, basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            20 * 1024 * 1024, 1, 401, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(
            args, metadata, hdf5_params,
            lambda basis_file: np.random.default_rng(1234).random(
                (image_size, image_size, 3), dtype=np.float32
            )
        )

        # clean up, so that the function can be run multiple times in a row
        shutil.rmtree(subdir)

    def test_dummy_basis100_random_subset100_small_cache(
        self, benchmark, tmp_path
    ):
        benchmark(
            self.build_matrix_dummy_basis100_random_subset100_small_cache,
            tmp_path
        )
        assert True

    def build_matrix_dummy_basis100_subset100_small_cache(self, tmp_path):
        basis_size = 100
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        subdir = tmp_path.joinpath('dummy_basis10')
        subdir.mkdir()
        args = DummyArgs(subdir, '', '')
        files = [
            {'coord_x': x, 'coord_y': y}
            for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            for y in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ]

        assert len(files) == 100

        metadata = DummyMetadata(
            files, basis_size, basis_size, image_size, image_size,
            basis_size**2, image_size**2
        )
        hdf5_params = DummyHDF5Params(
            20 * 1024 * 1024, 1, 401, matrix_shape, chunk_shape
        )

        build_matrix.build_matrix(
            args, metadata, hdf5_params,
            lambda basis_file: np.random.default_rng(1234).random(
                (image_size, image_size, 3), dtype=np.float32
            )
        )

        # clean up, so that the function can be run multiple times in a row
        shutil.rmtree(subdir)

    def test_dummy_basis100_subset100_small_cache(self, benchmark, tmp_path):
        benchmark(
            self.build_matrix_dummy_basis100_subset100_small_cache, tmp_path
        )
        assert True

    def build_matrix_files_basis10(self, tmp_path):
        basis_size = 10
        image_size = 640
        matrix_shape = (image_size, image_size, 3, basis_size**2)
        chunk_shape = (image_size, image_size, 3, 1)
        subdir = tmp_path.joinpath('files_basis10')
        subdir.mkdir()
        args = DummyArgs(subdir, 'scene.npy', '^([0-9]+)-([0-9]+)$')
        create_dummy_files(
            subdir,
            (image_size, image_size, 3),
            (basis_size, basis_size)
        )

        metadata = build_matrix.extract_metadata(args)
        hdf5_params = DummyHDF5Params(
            2000 * 1024 * 1024, 1, 40009, matrix_shape, chunk_shape
        )
        build_matrix.build_matrix(args, metadata, hdf5_params)

        # clean up, so that the function can be run multiple times in a row
        shutil.rmtree(subdir)

    def test_files_basis10(self, benchmark, tmp_path):
        benchmark(self.build_matrix_files_basis10, tmp_path)
        assert True
