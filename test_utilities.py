import pytest
import math

import utilities


class TestHDF5Params:
    def test_chunk_shape_insert_dim1(self):
        matrix_shape = (250, 10, 3)
        insert_dim = 1

        actual = utilities.HDF5Params(matrix_shape, insert_dim)
        expected = (250, 1, 3)

        assert actual.chunk_shape == expected

    def test_chunk_shape_insert_dim0(self):
        matrix_shape = (250, 10, 3)
        insert_dim = 0

        actual = utilities.HDF5Params(matrix_shape, insert_dim)
        expected = (1, 10, 3)

        assert actual.chunk_shape == expected

    def test_chunk_size(self):
        matrix_shape = (250, 10, 3)
        insert_dim = 1

        actual = utilities.HDF5Params(matrix_shape, insert_dim)
        expected = 250 * 1 * 3 * 4

        assert actual.chunk_size == expected

    def test_rdcc_nbytes(self):
        matrix_shape = (250, 10, 3)
        insert_dim = 1

        actual = utilities.HDF5Params(
            matrix_shape, insert_dim, cache_size=10
        )
        expected = 10 * 1024 * 1024

        assert actual.rdcc_nbytes == expected

    def test_rdcc_nslots(self):
        matrix_shape = (250, 10, 3)
        insert_dim = 1

        actual = utilities.HDF5Params(
            matrix_shape, insert_dim, cache_size=10
        )
        expected = math.ceil((10 * 1024 * 1024) / (250 * 1 * 3 * 4)) * 100

        assert actual.rdcc_nslots == expected

    def test_cache_too_small(self):
        matrix_shape = (100*1000*1000, 10, 3)
        insert_dim = 1

        with pytest.raises(ValueError):
            utilities.HDF5Params(
                matrix_shape, insert_dim, cache_size=1
            )

    def test_cache_larger_than_available_memory(self):
        matrix_shape = (250, 10, 3)
        insert_dim = 1

        with pytest.raises(ValueError):
            utilities.HDF5Params(
                matrix_shape, insert_dim, cache_size=10000
            )
