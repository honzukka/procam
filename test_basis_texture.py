import numpy as np

import basis_texture


class DummyArgs:
    def __init__(self, res, pos):
        self.res = res
        self.pos = pos


def test_unit_texture():
    args = DummyArgs((1, 1), (1, 1))
    actual = basis_texture.generate_basis_texture(args)
    expected = np.ones((1, 1, 3))

    assert np.array_equal(actual, expected)


def test_last_index():
    args = DummyArgs((5, 5), (5, 5))
    actual = basis_texture.generate_basis_texture(args)
    expected = np.zeros((5, 5, 3))
    expected[4, 4, :] = 1.0

    assert np.array_equal(actual, expected)


def test_rectangular():
    args = DummyArgs((10, 5), (8, 3))
    actual = basis_texture.generate_basis_texture(args)
    expected = np.zeros((5, 10, 3))
    expected[2, 7] = 1.0

    assert np.array_equal(actual, expected)


def test_index_outside1():
    args = DummyArgs((5, 5), (6, 1))
    actual = basis_texture.generate_basis_texture(args)
    expected = None

    assert actual == expected


def test_index_outside2():
    args = DummyArgs((5, 5), (1, 6))
    actual = basis_texture.generate_basis_texture(args)
    expected = None

    assert actual == expected


def test_invalid_resolution_x():
    args = DummyArgs((0, 5), (3, 1))
    actual = basis_texture.generate_basis_texture(args)
    expected = None

    assert actual == expected


def test_invalid_resolution_y():
    args = DummyArgs((5, 0), (3, 1))
    actual = basis_texture.generate_basis_texture(args)
    expected = None

    assert actual == expected


def test_invalid_position_x():
    args = DummyArgs((5, 5), (0, 1))
    actual = basis_texture.generate_basis_texture(args)
    expected = None

    assert actual == expected


def test_invalid_position_y():
    args = DummyArgs((5, 5), (1, 0))
    actual = basis_texture.generate_basis_texture(args)
    expected = None

    assert actual == expected
