import keras.ops as ops


def L2(x):

    return ops.sqrt(ops.sum(x**2))
