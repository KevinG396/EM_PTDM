
import tensorflow as tf
import jax.numpy as jnp
import numpy as np
import jax
import keras.src.backend as backend
from typing import Optional, Sequence, Tuple, Union


def ifft2c(data, norm="ortho"):
    if backend.backend() == "tensorflow":
        return ifft2c_tf(data, norm)
    elif backend.backend() == "jax":
        return ifft2c_jax(data, norm)


def fft2c(data, norm="ortho"):
    if backend.backend() == "tensorflow":
        return fft2c_tf(data, norm)
    elif backend.backend() == "jax":
        return fft2c_jax(data, norm)


def ifftshift(data, axes):
    if backend.backend() == "tensorflow":
        return tf.signal.ifftshift(data, axes=axes)
    elif backend.backend() == "jax":
        return jnp.fft.ifftshift(data, axes=axes)


def fftshift(data, axes):
    if backend.backend() == "tensorflow":
        return tf.signal.fftshift(data, axes=axes)
    elif backend.backend() == "jax":
        return jnp.fft.fftshift(data, axes=axes)


def complex_abs(data):
    if backend.backend() == "tensorflow":
        return complex_abs_tf(data)
    elif backend.backend() == "jax":
        return complex_abs_jax(data)


def view_as_complex_jax(data):

    assert data.shape[-1] == 2
    return jax.lax.complex(data[..., -2], data[..., -1])


def view_as_real_jax(data):

    real_part = jnp.real(data)
    imaginary_part = jnp.imag(data)
    return jnp.stack([real_part, imaginary_part], axis=-1)


def fft2c_jax(data, norm: str = "ortho"):

    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = jnp.fft.ifftshift(data, axes=[-3, -2])
    data = view_as_real_jax(
        jnp.fft.fftn(view_as_complex_jax(data), axes=(-2, -1), norm=norm)
    )
    # data = jnp.fft.fftshift(data, axes=[-3, -2])

    return data


def fft2c_tf(data, norm: str = "ortho"):

    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = tf.signal.ifftshift(data, axes=[-3, -2])
    data = view_as_complex_tf(data)
    data = view_as_real_tf(
        tf.signal.fftnd(data, data.shape[-2:], axes=(-2, -1), norm=norm)
    )
    # data = tf.signal.fftshift(data, axes=[-3, -2])

    return data


def complex_abs_tf(data):

    assert data.shape[-1] == 2
    return tf.sqrt(tf.reduce_sum(data**2, axis=-1))


def complex_abs_jax(data):

    assert data.shape[-1] == 2
    return jnp.sqrt(jnp.sum(data**2, axis=-1))


def view_as_real_tf(data):

    real_part = tf.math.real(data)
    imaginary_part = tf.math.imag(data)
    return tf.stack([real_part, imaginary_part], axis=-1)


def view_as_complex_tf(data):

    assert data.shape[-1] == 2
    return tf.complex(data[..., -2], data[..., -1])


def ifft2c_jax(data, norm="ortho"):

    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = jnp.fft.ifftshift(data, axes=[-3, -2])
    data = view_as_real_jax(
        jnp.fft.ifftn(view_as_complex_jax(data), axes=(-2, -1), norm=norm)
    )
    data = jnp.fft.fftshift(data, axes=[-3, -2])

    return data


def ifft2c_tf(data, norm="ortho"):

    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = tf.signal.ifftshift(data, axes=[-3, -2])
    data = view_as_real_tf(
        tf.signal.ifftnd(view_as_complex_tf(data), axes=(-2, -1), norm=norm)
    )
    data = tf.signal.fftshift(data, axes=[-3, -2])

    return data


class MaskFunc:


    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        seed: int = None,
        allow_any_combination: bool = False,
    ):
 
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng_key = jax.random.key(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ):

        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        center_mask, accel_mask, num_low_frequencies = self.sample_mask(shape, offset)

        # combine masks together
        return (
            jnp.logical_or(center_mask, accel_mask).astype(jnp.float32),
            num_low_frequencies,
        )

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ):

        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask, shape: Sequence[int]):
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return jnp.array(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:

        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return jax.random.choice(
                self.rng_key, self.center_fractions
            ), jax.random.choice(self.rng_key, self.accelerations)
        else:
            choice = int(
                jax.random.randint(
                    self.rng_key,
                    minval=0,
                    maxval=len(self.center_fractions),
                    shape=(1,),
                )
            )
            return self.center_fractions[choice], self.accelerations[choice]


class RandomMaskFunc(MaskFunc):


    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return jax.random.uniform(self.rng_key, shape=(num_cols,)) < prob


def apply_mask(
    data,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
):

    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies
