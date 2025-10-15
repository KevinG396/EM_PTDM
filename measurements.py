
import abc

import numpy as np
from jax import tree_util
from keras import ops

from utils import mri
from utils.keras_utils import check_keras_backend

check_keras_backend()

_OPERATORS = {}


def register_operator(cls=None, *, name=None):

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _OPERATORS:
            raise ValueError(f"Already registered operator with name: {local_name}")
        _OPERATORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_operator(name):
    assert (
        name in _OPERATORS
    ), f"Operator {name} not found. Available operators: {_OPERATORS.keys()}"
    return _OPERATORS[name]


class LinearOperator(abc.ABC):
    """Linear operator class y = Ax + n."""

    sigma = 0.0

    @abc.abstractmethod
    def forward(self, data):
        """Implements the forward operator A: x -> y."""
        raise NotImplementedError

    @abc.abstractmethod
    def corrupt(self, data):
        """Corrupt the data. Similar to forward but with noise."""
        raise NotImplementedError

    @abc.abstractmethod
    def transpose(self, data):
        """Implements the transpose operator A^T: y -> x."""
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        """String representation of the operator."""
        raise NotImplementedError

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children)

    def _tree_flatten(self):
        return (), ()


@register_operator(name="inpainting")
class InpaintingOperator(LinearOperator):
    """Inpainting operator A = I * M."""

    def __init__(self, mask, block_mask):
        self.mask = mask
        self.block_mask = block_mask

    def forward(self, data):
        return data * self.mask

    def corrupt(self, data):
        return self.forward(data)

    def transpose(self, data):
        return data * self.mask

    def __str__(self):
        return "y = Ax + n, where A = I * M"

    def _tree_flatten(self):
        return (self.mask,self.block_mask), ()

@register_operator(name="block")
class BlockOperator(LinearOperator):
    """Inpainting operator A = I * M."""

    def __init__(self, mask,block_mask):
        self.mask = mask
        self.block_mask = block_mask

    def forward(self, data):
        return data * self.mask

    def corrupt(self, data):
        return self.forward(data)

    def transpose(self, data):
        return data * self.mask

    def __str__(self):
        return "y = Ax + n, where A = I * M"

    def _tree_flatten(self):
        return (self.mask,), ()

@register_operator(name="fourier")
class FourierOperator(LinearOperator):
    """Fourier operator A = F."""

    def forward(self, data):
        return mri.fft2c(data)

    def corrupt(self, data):
        return mri.fft2c(data)

    def transpose(self, data):
        raise mri.ifft2c(data)

    def __str__(self):
        return "y = F(x)"


@register_operator(name="masked_fourier")
class MaskedFourierOperator(LinearOperator):

    def __init__(self, mask):
        self.mask = mask

    def forward(self, data):
        return self.mask * mri.fft2c(data)

    def corrupt(self, data):
        return self.mask * mri.fft2c(data)

    def transpose(self, data):
        raise self.mask * mri.ifft2c(data)

    def __str__(self):
        return "y = M*F(x)"

    def _tree_flatten(self):
        return (self.mask,), ()


def prepare_measurement(operator_name, target_imgs, **measurement_kwargs):

    operator = get_operator(operator_name)

    # set defaults for each operator --  configurable by changing operator.mask
    #print(f"measurement_kwargs: {measurement_kwargs}")
    if not measurement_kwargs:
        if operator_name == "masked_fourier":
            # default to a centered 4x acceleration mask.
            mask = (
                ops.zeros_like(target_imgs.shape[1:]).at[0, 32 + 16 : 64 + 16, 0].set(1)
            )
            measurement_kwargs = {"mask": mask}
        elif operator_name == "inpainting":
            # default to a mask hiding half pixels in the image.
            # Build measurement mask
            image_shape = target_imgs.shape[1:]
            mask = np.zeros(image_shape, dtype="float32")
            # mask out random half of pixels of the image
            n_total_samples = image_shape[0] * image_shape[1]
            random_idx = np.random.choice(
                n_total_samples, size=n_total_samples // 2, replace=False
            )
            random_idx = np.unravel_index(random_idx, image_shape[:-1])
            mask[random_idx] = 1
            mask = mask[None, :, :]  
            measurement_kwargs = {"mask": mask}
        elif operator_name == "block":

            block_size = 4
            image_shape = target_imgs.shape[1:]

            mask = np.zeros(image_shape, dtype="float32")
            block_mask = np.zeros((8,8), dtype="float32")

            n_total_samples = (image_shape[0]/4) * (image_shape[1]/4)
            random_idx = np.random.choice(
                n_total_samples, size=n_total_samples // 2, replace=False
            )
            random_idx = np.unravel_index(random_idx, image_shape[:-1])
            mask[random_idx] = 1
            mask = mask[None, :, :] 
            measurement_kwargs = {"mask": mask}
        else:
            raise ValueError(f"Operator `{operator_name}` not recognised.")


    operator = operator(**measurement_kwargs)

    measurements = operator.corrupt(target_imgs)
    return operator, measurements

for cls in LinearOperator.__subclasses__():
    tree_util.register_pytree_node(
        cls,
        cls._tree_flatten,
        cls._tree_unflatten,
    )

