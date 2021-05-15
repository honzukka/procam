import argparse
import sys

import numpy as np  # type: ignore

import utilities


def main():
    args = parse_arguments()

    basis_texture = generate_basis_texture(args)
    success = utilities.write_image(
        args.path,
        basis_texture,
        np.float16
    )

    if success:
        sys.exit(0)
    else:
        sys.exit(
            'Basis texture (res: [{}, {}], pos: [{}, {}]) not written!'
            .format(
                args.res[0], args.res[1],
                args.pos[0], args.pos[1]
            )
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--res',
        type=int,
        nargs=2,
        metavar=('x', 'y'),
        default=[3, 3],
        help='Basis texture resolution.'
    )

    parser.add_argument(
        '--pos',
        type=int,
        nargs=2,
        metavar=('a', 'b'),
        default=[1, 1],
        help='Coordinates of the lit up pixel starting from 1.'
    )

    parser.add_argument(
        'path',
        help=(
            'Output folder together with output file name. '
            'Since OpenImageIO is used to write the textures, '
            'the file extension influences the output format.'
        )
    )

    return parser.parse_args()


def generate_basis_texture(args):
    '''
    Returns an image of resolution args.res which has zero values
    everywhere except for index args.pos. This makes the image
    a canonical basis vector in a light transport matrix space.

    :param args: Objects containing fields res (texture resolution) \
                 and pos (lit up pixel index)

    :returns: float16 numpy array, None if pos is outside of the texture \
              or if resolution is < 1
    '''
    if (
        args.pos[0] > args.res[0] or
        args.pos[1] > args.res[1] or
        min(args.res[0], min(args.res[1], min(args.pos[0], args.pos[1]))) < 1
    ):
        return None

    basis_texture = np.zeros(
        (args.res[1], args.res[0], 3),
        dtype=np.float16
    )
    basis_texture[args.pos[1] - 1, args.pos[0] - 1, :] = 1.0

    return basis_texture


if __name__ == "__main__":
    main()
