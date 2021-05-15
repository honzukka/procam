from __future__ import annotations
import os
import math
from typing import List, Tuple, Optional
import pickle
import sys
import imp
import inspect
import importlib

import psutil               # type: ignore
import h5py                 # type: ignore
import numpy as np          # type: ignore
import OpenImageIO as oiio  # type: ignore
import ffmpeg               # type: ignore
import torch
import PIL.Image            # type: ignore

import constants


class Metadata:
    '''
    Description of the data in the light transport matrix file.
    '''
    def __init__(
        self, files: Optional[List[dict]], basis_width: int, basis_height: int,
        image_width: int, image_height: int,
        extra_illum: Optional[np.ndarray]
    ):
        if files is None:
            self.files = [{}]   # type: List[dict]
        else:
            self.files = files
        self.extra_illum = extra_illum

        self.basis_width = basis_width
        self.basis_height = basis_height
        self.image_width = image_width
        self.image_height = image_height

        # derived quantities
        self.n_bases = basis_width * basis_height
        self.n_pixels = image_width * image_height

    def __str__(self) -> str:
        return ('{{[{}, {}] [{}, {}]}}'.format(
            self.basis_width, self.basis_height,
            self.image_width, self.image_height
        ))

    def get_numpy_array(self) -> np.ndarray:
        return np.array(
            [
                self.basis_width,
                self.basis_height,
                self.image_width,
                self.image_height
            ], dtype=np.int32
        )

    @staticmethod
    def create_from_array(array: np.ndarray) -> Metadata:
        return Metadata(None, array[0], array[1], array[2], array[3], None)

    @staticmethod
    def create_from_matrix_file(path: str) -> Metadata:
        with h5py.File(path, 'r') as matrix_file:
            return Metadata.create_from_array(
                matrix_file[constants.METADATA_DATASET_NAME]
            )


class HDF5Params:
    '''
    Parameters for reading and writing HDF5 files.
    '''
    def __init__(
        self, matrix_shape: Tuple[int, int, int, int], insert_dim: int = 3,
        cache_size: int = 100, dtype: type = np.float32,
        rdcc_nslots: int = -1, quiet: bool = False
    ):
        self.rdcc_nbytes = cache_size * 1024 * 1024
        self.rdcc_w0 = 1

        if psutil.virtual_memory().available < self.rdcc_nbytes:
            raise ValueError(
                'Requested HDF5 cache size is larger than available memory!'
            )

        w_size = np.dtype(dtype).itemsize
        self.matrix_shape = matrix_shape
        self.matrix_size = np.prod(self.matrix_shape, dtype=np.uint64) * w_size
        chunk_shape = [x for x in matrix_shape]    # tuple to list
        chunk_shape[insert_dim] = 1
        self.chunk_shape = tuple(chunk_shape)      # list to tuple
        self.chunk_size = np.prod(self.chunk_shape, dtype=np.uint64) * w_size

        if self.chunk_size > self.rdcc_nbytes:
            raise ValueError(
                'HDF5 chunk size is larger than requested cache size!'
            )

        self.n_chunks = matrix_shape[insert_dim]
        self.n_chunks_in_cache = min(
            math.floor(self.rdcc_nbytes / self.chunk_size), self.n_chunks
        )

        # setting this according to h5py docs
        # (http://docs.h5py.org/en/stable/high/file.html#chunk-cache)
        self.rdcc_nslots = (rdcc_nslots
                            if rdcc_nslots != -1
                            else math.ceil(
                                self.rdcc_nbytes / self.chunk_size) * 100
                            )

        if not quiet:
            print(
                (
                    'HDF5 parameters configured to be the following:\n'
                    '\trdcc_nbytes: {}\n'
                    '\trdcc_w0: {}\n'
                    '\trdcc_nslots: {}\n'
                    '\tmatrix_shape: {}\n'
                    '\tmatrix_size: {}\n'
                    '\tchunk_shape: {}\n'
                    '\tchunk_size: {}\n'
                ).format(
                    self.rdcc_nbytes, self.rdcc_w0, self.rdcc_nslots,
                    self.matrix_shape, self.matrix_size,
                    self.chunk_shape, self.chunk_size
                )
            )

    @staticmethod
    def create_from_metadata(
        metadata: Metadata, cache_size: int, rdcc_nslots: int = -1
    ) -> HDF5Params:
        return HDF5Params(
            (metadata.image_height, metadata.image_width, 3, metadata.n_bases),
            cache_size=cache_size, rdcc_nslots=rdcc_nslots
        )


def write_image(
    path: str, image: np.ndarray, number_fmt: type = np.float32
) -> bool:
    number_format = None
    if number_fmt == np.float16:
        number_format = oiio.HALF
    elif number_fmt == np.float32:
        number_format = oiio.FLOAT

    spec = oiio.ImageSpec(
        image.shape[1],
        image.shape[0],
        image.shape[2],
        number_format
    )
    output = oiio.ImageOutput.create(path)
    if output is None:
        return False

    success = output.open(path, spec)
    success &= output.write_image(image)
    success &= output.close()

    return success


def write_video(
    filename: str, frames: np.ndarray, framerate: int = 30
) -> bool:
    assert type(frames) is np.ndarray

    if (len(frames.shape) != 4):
        print('Video not written: Input does not have shape [F, H, W, C]!')
        return False

    assert (frames >= 0.0).all()
    if (frames > 1.0).any():
        print('write_video(): Clipping values that are larger than 1.0.')
        frames = np.clip(frames, None, 1.0)

    # frames need to be in uint8
    if (frames.dtype != np.uint8):
        frames = (frames * 255).astype(np.uint8)

    n, height, width, channels = frames.shape
    even_width = width if width % 2 == 0 else width + 1
    even_height = height if height % 2 == 0 else height + 1

    # create process and open input pipe
    process = (
        ffmpeg
        .input(
            'pipe:', format='rawvideo', pix_fmt='rgb24',
            s='{}x{}'.format(width, height), r=framerate
        )
        .filter('pad', width=even_width, height=even_height)
        .output(filename, pix_fmt='yuv420p', vcodec='libx264', r=framerate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # feed frame into the pipe
    for frame in frames:
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

    return True


def read_image(path: str, linearize: bool = True) -> np.ndarray:
    if not os.path.isfile(path):
        raise ValueError('File {} does not exist.'.format(path))

    image_no_alpha = oiio.ImageBuf(path).get_pixels()[:, :, 0:3]

    if not linearize:
        return image_no_alpha

    # linearize color space if image format is PNG or JPG
    _, ext = os.path.splitext(path)
    if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
        return np.power(image_no_alpha, 2.2)

    # otherwise assume color space is linear
    return image_no_alpha


def read_video(filename: str, target_width: int) -> np.ndarray:
    probe = ffmpeg.probe(filename)
    video_info = next(
        x for x in probe['streams'] if x['codec_type'] == 'video'
    )
    width = int(video_info['width'])
    height = int(video_info['height'])
    aspect = height / width
    target_height = int(aspect * target_width)

    out, _ = (
        ffmpeg
        .input(filename)
        .filter('scale', target_width, target_height)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )

    video = (
        np.frombuffer(out, np.uint8)
        .reshape((-1, target_height, target_width, 3))
        .astype(np.float32)
    )

    # normalize and linearize color space
    return np.power(video / 255.0, 2.2)


# assumes that input image is in linear color space
def tonemap(
    image: np.ndarray, reinhard: bool = True, gamma: float = 2.2
) -> np.ndarray:
    assert type(image) is np.ndarray

    if reinhard:
        image = image / (image + 1.0)
    return np.power(image, 1.0 / gamma)


def load_matrix_torch(
    path: str, hdf5_params: HDF5Params, start: int = 0,
    end: Optional[int] = None, matrix_name: Optional[str] = None
) -> torch.Tensor:
    return torch.from_numpy(
        load_matrix(path, hdf5_params, start, end, matrix_name)
    )


def load_matrix_wrap(path: str, cache_size: int = 1000) -> np.ndarray:
    metadata = Metadata.create_from_matrix_file(path)
    hdf5_params = HDF5Params.create_from_metadata(metadata, cache_size)
    return load_matrix(path, hdf5_params), load_extra_illum(path, hdf5_params)


def load_matrix(
    path: str, hdf5_params: HDF5Params, start: int = 0,
    end: Optional[int] = None, matrix_name: Optional[str] = None
) -> np.ndarray:
    with h5py.File(
        path, 'r', rdcc_nbytes=hdf5_params.rdcc_nbytes,
        rdcc_w0=hdf5_params.rdcc_w0, rdcc_nslots=hdf5_params.rdcc_nslots
    ) as matrix_file:
        return matrix_file[constants.MATRIX_DATASET_NAME][:, :, :, start:end]


def load_extra_illum(path: str, hdf5_params: HDF5Params):
    with h5py.File(
        path, 'r', rdcc_nbytes=hdf5_params.rdcc_nbytes,
        rdcc_w0=hdf5_params.rdcc_w0, rdcc_nslots=hdf5_params.rdcc_nslots
    ) as matrix_file:
        if constants.EXTRA_ILLUM_DATASET_NAME not in matrix_file:
            return None
        else:
            return matrix_file[constants.EXTRA_ILLUM_DATASET_NAME][()]


def get_animation_framerate(n_frames: int, target_seconds: int = 10) -> int:
    if n_frames <= 10:
        return 1

    return int(n_frames / target_seconds)


def save_model(model, path):
    """
    Saves the model(s), including the definitions in its containing module.
    Restore the model(s) with load_model. References to other modules
    are not chased; they're assumed to be available when calling load_model.
    The state of any other object in the module is not stored.
    Written by Pauli Kemppinen.
    """
    model_pickle = pickle.dumps(model)

    # Handle dicts, lists and tuples of models.
    model = list(model.values()) if isinstance(model, dict) else model
    model = (
        (model,)
        if not (isinstance(model, list) or isinstance(model, tuple))
        else model
    )

    # Create a dict of modules that maps from name to source code.
    module_names = {m.__class__.__module__ for m in model}
    modules = {
        name:
            inspect.getsource(importlib.import_module(name))
            for name in module_names
    }

    pickle.dump((modules, model_pickle), open(path, 'wb'))


def load_model(path):
    """
    Loads the model(s) stored by save_model.
    Written by Pauli Kemppinen.
    """
    modules, model_pickle = pickle.load(open(path, 'rb'))

    # Temporarily add or replace available modules with stored ones.
    sys_modules = {}
    for name, source in modules.items():
        module = imp.new_module(name)
        exec(source, module.__dict__)
        if name in sys.modules:
            sys_modules[name] = sys.modules[name]
        sys.modules[name] = module

    # Map pytorch models to cpu if cuda is not available.
    if imp.find_module('torch'):
        import torch
        original_load = torch.load

        def map_location_cpu(*args, **kwargs):
            kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)
        torch.load = (
            original_load
            if torch.cuda.is_available()
            else map_location_cpu
        )

    model = pickle.loads(model_pickle)

    if imp.find_module('torch'):
        torch.load = original_load  # Revert monkey patch.

    # Revert sys.modules to original state.
    for name in modules.keys():
        if name in sys_modules:
            sys.modules[name] = sys_modules[name]
        else:
            # Just to make sure nobody else depends on these existing.
            sys.modules.pop(name)

    return model


# compute the Gram matrix of a batch of activations (of a batch of images)
def gram_matrix(
    activations1: torch.Tensor, activations2: torch.Tensor,
    offset: float = -1.0
) -> torch.Tensor:
    assert activations1.shape[0] == activations2.shape[0]
    assert activations1.shape[2:] == activations2.shape[2:]

    activations1 = activations1 + offset
    activations2 = activations2 + offset

    b, n1, x, y = activations1.size()
    _, n2, _, _ = activations2.size()
    activation_matrix1 = activations1.view(b, n1, x * y)
    activation_matrix2 = activations2.view(b, n2, x * y)
    G = torch.bmm(activation_matrix1, activation_matrix2.transpose(1, 2))
    return G.sum(dim=0).div(b * (n1 + n2) * x * y)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def inv_sigmoid(y: torch.Tensor) -> torch.Tensor:
    return -torch.log((1.0 / y) - 1.0)


def open_hdf5_file(
    path, mode, chunk_shape, insert_dim, cache_size=100, dtype=np.float32
):
    hdf5_params = HDF5Params(chunk_shape, insert_dim, cache_size, quiet=True)
    return h5py.File(
        path, mode, rdcc_nbytes=hdf5_params.rdcc_nbytes,
        rdcc_w0=hdf5_params.rdcc_w0, rdcc_nslots=hdf5_params.rdcc_nslots
    )


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.numpy())


def resize(
    img: np.ndarray, target_size: Tuple[int, int], resample: str
) -> np.ndarray:
    if resample == 'lanczos':
        pil_resample = PIL.Image.LANCZOS
    elif resample == 'nearest':
        pil_resample = PIL.Image.NEAREST
    else:
        raise ValueError('Unsupported resample value: {}'.format(resample))
    h, w = target_size
    img_pil = PIL.Image.fromarray(np.uint8(img * 255))

    img_pil_resized = img_pil.resize((w, h), pil_resample)
    img_numpy_resized = np.array(img_pil_resized, dtype=np.float32) / 255.0

    return img_numpy_resized


def get_crop_size(
    crop_arg: Optional[List[int]], default_height: int, default_width: int
) -> Optional[Tuple[int, int]]:
    crop = get_crop(crop_arg, default_height, default_width)
    return (crop[3] - crop[1], crop[2] - crop[0])


def get_crop(
    crop_arg: Optional[List[int]], default_height: int, default_width: int
) -> List[int]:
    return (
        crop_arg
        if crop_arg is not None
        else [0, 0, default_width, default_height]
    )
