
import functools

import jax
import keras
import numpy as np
import tensorflow as tf


class AutoGrad:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.function = None

        if verbose:
            print(f"Using backend: {self.backend}")

    @property
    def backend(self):
        return keras.backend.backend()

    @backend.setter
    def backend(self, backend):
        raise ValueError("Cannot change backend currently. Needs reimport of keras.")
        # keras.config.set_backend(backend)

    def set_function(self, function):
        self.function = function

    def gradient(self, variable, **kwargs):

        variable = keras.ops.convert_to_tensor(variable)
        if self.function is None:
            raise ValueError(
                "Function not set. Use `set_function` to set a custom function."
            )
        assert self.backend in [
            "torch",
            "tensorflow",
            "jax",
        ], f"Unsupported backend: {self.backend}"

        if self.backend == "torch":
            variable = variable.detach().requires_grad_(True)
            out = self.function(variable, **kwargs)
            gradients = torch.autograd.grad(out, variable)[0]
            return gradients
        elif self.backend == "tensorflow":
            with tf.GradientTape() as tape:
                tape.watch(variable)
                out = self.function(variable, **kwargs)
            gradients = tape.gradient(out, variable)
            return gradients
        elif self.backend == "jax":
            func = functools.partial(self.function, **kwargs)
            return jax.grad(func)(variable)

    def gradient_and_value(self, variable, has_aux: bool = False, **kwargs):

        variable = keras.ops.convert_to_tensor(variable)
        if self.function is None:
            raise ValueError(
                "Function not set. Use `set_function` to set a custom function."
            )
        assert self.backend in [
            "torch",
            "tensorflow",
            "jax",
        ], f"Unsupported backend: {self.backend}"

        if self.backend == "torch":
            variable = variable.detach().requires_grad_(True)
            if has_aux:
                out, aux = self.function(variable, **kwargs)
            else:
                out = self.function(variable, **kwargs)
            gradients = torch.autograd.grad(out, variable)[0]
        elif self.backend == "tensorflow":
            with tf.GradientTape() as tape:
                tape.watch(variable)
                if has_aux:
                    out, aux = self.function(variable, **kwargs)
                else:
                    out = self.function(variable, **kwargs)
            gradients = tape.gradient(out, variable)
        elif self.backend == "jax":
            out, gradients = jax.value_and_grad(
                self.function, argnums=0, has_aux=has_aux
            )(variable, **kwargs)
            if has_aux:
                out, aux = out

        if has_aux:
            return gradients, (out, aux)
        return gradients, out

    def get_gradient_jit_fn(self):
        if self.backend == "jax":
            return jax.jit(self.gradient)
        elif self.backend == "tensorflow":
            return tf.function(self.gradient, jit_compile=True)
        elif self.backend == "torch":
            return torch.compile(self.gradient)

    def get_gradient_and_value_jit_fn(self, has_aux: bool = False, disable_jit=False):
        if self.backend == "jax":
            if disable_jit:
                return lambda x, **kwargs: self.gradient_and_value(
                    x, has_aux=has_aux, **kwargs
                )
            return jax.jit(
                lambda x, **kwargs: self.gradient_and_value(
                    x, has_aux=has_aux, **kwargs
                )
            )

        elif self.backend == "tensorflow":
            return tf.function(
                lambda x, **kwargs: self.gradient_and_value(
                    x, has_aux=has_aux, **kwargs
                )
            )
        elif self.backend == "torch":
            raise NotImplementedError("Jitting not supported for torch backend.")
