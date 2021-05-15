from typing import List
from argparse import Namespace
import os

import numpy as np                      # type: ignore
import h5py                             # type: ignore
from matplotlib import pyplot as plt    # type: ignore

import utilities
from loader import Data


class Logger:
    def __init__(self, data: Data, config: Namespace):
        self.checkpoint_every = config.checkpoint_every
        self.full_output = config.full_output
        self.target_ind = config.target_ind
        self.scene_ind = config.scene_ind

        self.target_img = data.target_imgs[self.target_ind]
        self.background_img = data.background_img

        self.scene_target_str = '{:03d}-{:03d}-{}-{}{}'.format(
            self.scene_ind,
            self.target_ind,
            config.scene_names[self.scene_ind],
            config.target_names[self.target_ind],
            config.suffix if hasattr(config, 'suffix') else ''
        )
        self.out_dir = os.path.join(config.out_dir, self.scene_target_str)
        os.makedirs(self.out_dir, exist_ok=True)

        self.losses = []            # type: List[float]
        self.brightness_vals = []   # type: List[float]
        self.projection_means = []  # type: List[np.ndarray]
        self.projection_mins = []   # type: List[np.ndarray]
        self.projection_maxs = []   # type: List[np.ndarray]

        if self.full_output:
            self.opt_images = []        # type: List[np.ndarray]
            self.projections = []       # type: List[np.ndarray]

    def log(
        self, loss: float, opt_image: np.ndarray, brightness: float,
        projection: np.ndarray
    ):
        self.losses.append(loss)
        self.brightness_vals.append(brightness)
        self.projection_means.append(
            np.mean(projection, axis=(0, 1, 2))
        )
        self.projection_mins.append(
            np.min(projection, axis=(0, 1, 2))
        )
        self.projection_maxs.append(
            np.max(projection, axis=(0, 1, 2))
        )

        if self.full_output:
            self.opt_images.append(opt_image)
            self.projections.append(projection)

    def save(self, *results):
        self.output_results(*results)
        self.output_checkpoints()
        self.output_target_and_bg()

    def output_results(
        self, result_im: np.ndarray, result_proj: np.ndarray,
        result_brightness: np.ndarray,
        initial_im: np.ndarray, initial_proj: np.ndarray
    ):
        for i in range(result_im.shape[0]):
            utilities.write_image(
                os.path.join(self.out_dir, 'output_im_{}.exr'.format(i)),
                result_im[i]
            )
            utilities.write_image(
                os.path.join(self.out_dir, 'output_proj_{}.exr'.format(i)),
                result_proj[i]
            )
            utilities.write_image(
                os.path.join(self.out_dir, 'init_im_{}.exr'.format(i)),
                initial_im[i]
            )
            utilities.write_image(
                os.path.join(self.out_dir, 'init_proj_{}.exr'.format(i)),
                initial_proj[i]
            )
        with open(
            os.path.join(self.out_dir, 'final_brightness.out'), 'w'
        ) as f:
            f.write('{}'.format(result_brightness))

    def output_checkpoints(self):
        if len(self.losses) == 0:
            return
        self.lists_to_numpy()

        # prepare plots from some of the data
        self.plot_loss(self.out_dir)
        self.plot_values(self.out_dir)

        if not self.full_output:
            return

        # write all logged data to an HDF5 file
        with h5py.File(os.path.join(self.out_dir, 'log.hdf5'), mode='w') as f:
            f.create_dataset(
                'checkpoint_period', shape=(), dtype=np.uint32,
                data=np.array(self.checkpoint_every)
            )
            f.create_dataset('loss', data=self.losses_numpy)
            f.create_dataset('brightness', data=self.brightness_vals_numpy)
            f.create_dataset('proj_means', data=self.projection_means_numpy)
            f.create_dataset('proj_mins', data=self.projection_mins_numpy)
            f.create_dataset('proj_maxs', data=self.projection_maxs_numpy)

        # to ensure write speed is optimal for large data, open the file again
        # with a suitable chunk size
        with utilities.open_hdf5_file(
            os.path.join(self.out_dir, 'log.hdf5'), 'a',
            self.opt_images_numpy.shape, 0
        ) as f:
            f.create_dataset(
                'opt_images', data=self.opt_images_numpy,
                compression='gzip', compression_opts=9
            )

        with utilities.open_hdf5_file(
            os.path.join(self.out_dir, 'log.hdf5'), 'a',
            self.projections_numpy.shape, 0
        ) as f:
            f.create_dataset(
                'proj_images', data=self.projections_numpy,
                compression='gzip', compression_opts=9
            )

    def output_target_and_bg(self):
        utilities.write_image(
            os.path.join(self.out_dir, 'target.exr'), self.target_img
        )

        if self.background_img is not None:
            utilities.write_image(
                os.path.join(self.out_dir, 'bg.exr'), self.background_img
            )

    def plot_loss(self, out_dir):
        x = self.get_x_axis()
        plt.plot(x, self.losses)
        plt.yscale('log')
        plt.savefig(os.path.join(out_dir, 'loss_plot.png'))
        plt.close()

    def plot_values(self, out_dir):
        x = self.get_x_axis()
        projection_means_numpy = np.stack(self.projection_means)
        projection_mins_numpy = np.stack(self.projection_mins)
        projection_maxs_numpy = np.stack(self.projection_maxs)
        plt.plot(
            x, self.brightness_vals, color='xkcd:black', label='brightness'
        )
        plt.plot(
            x, projection_means_numpy[:, 0], color='xkcd:red',
            linewidth=3, label='proj_mean'
        )
        plt.plot(x, projection_mins_numpy[:, 0], color='xkcd:red', linewidth=1)
        plt.plot(x, projection_maxs_numpy[:, 0], color='xkcd:red', linewidth=1)
        plt.fill_between(
            x, projection_mins_numpy[:, 0], projection_maxs_numpy[:, 0],
            facecolor='xkcd:peach', label='proj_vals'
        )
        plt.xlabel('steps')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'value_plot.png'))
        plt.close()

    def get_x_axis(self):
        return list(range(
            self.checkpoint_every,
            len(self.losses) * self.checkpoint_every + 1,
            self.checkpoint_every
        ))

    def lists_to_numpy(self):
        self.losses_numpy = (
            np.stack(self.losses).astype(np.float32)
        )
        self.brightness_vals_numpy = (
            np.stack(self.brightness_vals).astype(np.float32)
        )
        self.projection_means_numpy = (
            np.stack(self.projection_means).astype(np.float32)
        )
        self.projection_mins_numpy = (
            np.stack(self.projection_mins).astype(np.float32)
        )
        self.projection_maxs_numpy = (
            np.stack(self.projection_maxs).astype(np.float32)
        )

        if self.full_output:
            self.opt_images_numpy = (
                np.stack(self.opt_images).astype(np.float32)
            )
            self.projections_numpy = (
                np.stack(self.projections).astype(np.float32)
            )
