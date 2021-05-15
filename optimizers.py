from __future__ import annotations
import time
from argparse import Namespace
from typing import List, Tuple
from functools import reduce

import numpy as np          # type: ignore
import torch
from scipy import optimize  # type: ignore

import utilities
import models
from log import Logger


class OptimizerSciPy:
    def __init__(
        self, model: models.SynthesisModel, logger: Logger, config: Namespace
    ):
        self.model = model
        self.device = model.device
        self.logger = logger
        self.n_steps = config.n_steps
        self.checkpoint_every = config.checkpoint_every

        self.opt_image, self.opt_brightness = self.get_initial_guess()
        self.initial_guess = self.opt_image.clone()

        self.last_checkpoint_time = time.time()

    def optimize(self) -> torch.Tensor:
        wrapper = PyTorchModelWrapper(
            self.model,
            [self.opt_image.requires_grad_()],
            [self.opt_brightness],
            self.device
        )

        self.step = -1
        _ = optimize.minimize(
            wrapper.fun, wrapper.x0, method='L-BFGS-B', jac=wrapper.jac,
            bounds=optimize.Bounds(0.0, 1.0), callback=self.checkpoint,
            options={
                'maxiter': self.n_steps, 'maxcor': 20,
                'ftol': 0, 'gtol': 0,
            }
        )

        return self.output_results()

    def checkpoint(self, xk: np.ndarray):
        self.step += 1
        # TODO: SciPy optimizers support this, no need to emulate it here
        if self.step % self.checkpoint_every != self.checkpoint_every - 1:
            return

        time_delta = time.time() - self.last_checkpoint_time
        opt_image = self.opt_image.clone().detach()
        opt_brightness = self.opt_brightness.clone().detach()
        loss = self.model((opt_image, opt_brightness)).item()

        print('step: {}, loss: {} ({:.2f}s)'.format(
            self.step, loss, time_delta
        ))

        self.logger.log(
            loss,
            utilities.tensor_to_numpy(opt_image.cpu()),
            opt_brightness.item(),
            utilities.tensor_to_numpy(
                self.model.render((opt_image, opt_brightness)).cpu()
            )
        )

        self.last_checkpoint_time = time.time()

    def get_initial_guess(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.get_initial_guess()

    def output_results(self):
        opt_image = self.opt_image.detach()
        opt_brightness = self.opt_brightness.detach()

        final_image = utilities.tensor_to_numpy(
            opt_image.cpu()
        )
        final_projection = utilities.tensor_to_numpy(
            self.model.render((opt_image, opt_brightness))
            .detach().cpu()
        )
        final_brightness = opt_brightness.cpu().item()

        initial_image = utilities.tensor_to_numpy(
            self.initial_guess.cpu()
        )
        initial_projection = utilities.tensor_to_numpy(
            self.model.render((self.initial_guess, opt_brightness))
            .detach().cpu()
        )

        return (
            final_image, final_projection, final_brightness,
            initial_image, initial_projection
        )


class PyTorchModelWrapper:
    '''
    Wraps a PyTorch model, so that it can be used in a SciPy optimizer.
    Adapted: https://gist.github.com/gngdb/a9f912df362a85b37c730154ef3c294b
    '''
    def __init__(
        self, model: models.SynthesisModel, opt_vars: List[torch.Tensor],
        non_opt_vars: List[torch.Tensor], device: torch.device
    ):
        self.model = model
        self.device = device
        self.opt_vars = opt_vars
        self.non_opt_vars = non_opt_vars

        # store opt_vars shapes because SciPy only works with 1D arrays
        self.opt_var_shapes = [opt_var.size() for opt_var in opt_vars]

        # convert opt_vars to x0 (SciPy's initial guess)
        self.x0 = np.concatenate(
            [
                opt_var.detach().cpu().numpy().ravel().astype(np.float64)
                for opt_var in opt_vars
            ]
        )

    def unpack_opt_vars(self, x):
        x_ind = 0
        for i, shape in enumerate(self.opt_var_shapes):
            # unpack opt_var from 1D numpy array
            if shape == ():
                opt_var = x[x_ind:x_ind + 1]
            else:
                opt_var_length = reduce(lambda x, y: x * y, shape)
                opt_var = x[x_ind:x_ind + opt_var_length]
                opt_var = opt_var.reshape(*shape)

            # update the original opt_vars which are on the correct device
            # and have gradients enabled
            self.opt_vars[i].data = (
                torch.from_numpy(opt_var)
                .to(torch.float32)
                .to(self.device)
            )

            x_ind += opt_var_length

    def pack_grads(self):
        grads = []      # type: List[np.ndarray]
        for opt_var in self.opt_vars:
            grad = opt_var.grad.data.cpu().numpy().astype(np.float64)
            grads.append(grad.ravel())
        return np.concatenate(grads)

    def is_new(self, x):
        # if this is the first thing we've seen
        if not hasattr(self, 'cached_x'):
            return True
        else:
            # compare x to cached_x to see if we've been given a new input
            x, self.cached_x = np.array(x), np.array(self.cached_x)
            error = np.abs(x - self.cached_x)
            return error.max() > 1e-8

    def cache(self, x):
        # unpack x and load into opt_vars
        self.unpack_opt_vars(x)

        # store the raw array as well
        self.cached_x = x

        # zero the gradient
        for opt_var in self.opt_vars:
            if opt_var.grad is not None:
                opt_var.grad.data.zero_()

        # calculate the loss using the new opt_vars
        loss = self.model(tuple(self.opt_vars + self.non_opt_vars))

        # backprop the loss
        loss.backward()
        self.cached_f = loss.item()
        self.cached_jac = self.pack_grads()

    def fun(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_f

    def jac(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_jac
