#import jax
import os

import tensorflow as tf
from keras import ops
from keras.src import backend
import numpy as np
from autograd_backend import AutoGrad
from guidance.utils import L2
from utils.keras_utils import check_keras_backend
import jax.numpy as jnp
check_keras_backend()

_GUIDANCE = {}
def save_image_grid(images, save_path, grid_size=(4, 4)):

    images = np.array(images)
    B, H, W = images.shape[:3]
    assert B == grid_size[0] * grid_size[1], f"{B} mismatch {grid_size} "

    images = images.squeeze(-1) if images.ndim == 4 else images 
    grid_h = H * grid_size[0]
    grid_w = W * grid_size[1]
    canvas = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        canvas[row * H:(row + 1) * H, col * W:(col + 1) * W] = image

    Image.fromarray(canvas, mode="L").save(save_path)
    print(f"Saved: {save_path}")


def register_guidance(cls=None, *, name=None):

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _GUIDANCE:
            raise ValueError(f"Already registered guidance with name: {local_name}")
        _GUIDANCE[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_guidance(name):
    assert (
        name in _GUIDANCE
    ), f"Guidance {name} not found. Available guidance: {_GUIDANCE.keys()}"
    return _GUIDANCE[name]


@register_guidance(name="dps")
def get_dps(dm):

    autograd = AutoGrad()

    def compute_measurement_error(
        noisy_images, measurement, operator, omega, noise_rates, signal_rates
    ):
        pred_noises, pred_images = dm.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        measurement_error = omega * L2(measurement - operator.forward(pred_images))
        return measurement_error, (pred_noises, pred_images)

    autograd.set_function(compute_measurement_error)
    return autograd.get_gradient_and_value_jit_fn(has_aux=True)
