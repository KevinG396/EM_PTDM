import re
import os
import gc
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import psutil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import keras
import keras.ops as ops
from keras.callbacks import Callback

import utils.mri as mri
from utils.lib import log

class ClearMemory(Callback):

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()


def check_keras_backend():
    backend = os.environ.get("KERAS_BACKEND")
    assert backend == keras.backend.backend(), (
        f"Keras backend {keras.backend.backend()} does not match "
        f"environment variable KERAS_BACKEND {backend}"
    )
    if backend is None:
        raise ValueError("Please set the KERAS_BACKEND environment variable.")
    return backend


loss_funcs = {
    "mae": keras.losses.mean_absolute_error,
    "mse": keras.losses.mean_squared_error,
}


def get_loss_func(loss_name):
    if loss_name in loss_funcs:
        return loss_funcs[loss_name]
    else:
        raise ValueError(f"Loss function {loss_name} not found.")


def translate(array, range_from, range_to):

    left_min, left_max = range_from
    right_min, right_max = range_to
    assert left_min <= left_max, "boundaries are set incorrectly"
    assert right_min < right_max, "boundaries are set incorrectly"
    if left_min == left_max:
        return np.ones_like(array) * right_max

    value_scaled = (array - left_min) / (left_max - left_min)

    return right_min + (value_scaled * (right_max - right_min))


def get_normalization_layer(a, b, x_min=0, x_max=255):

    scale = (b - a) / (x_max - x_min)
    offset = a
    return keras.layers.Rescaling(scale=scale, offset=offset)


def grayscale_to_rgb(x):
    return keras.ops.repeat(x, 3, axis=-1)


def grayscale_to_random_rgb(images, min_val=None):


    if len(ops.shape(images)) == 3:
        images = ops.expand_dims(images, axis=0)

    if min_val is None:
        min_val = ops.min(images)

    n_channels = 3
    batch_size = ops.shape(images)[0]

    ch_indexes = keras.random.uniform(
        [batch_size],
        minval=0,
        maxval=n_channels,
    )
    ch_indexes = ops.cast(ch_indexes, dtype="int32")

    indices = ops.one_hot(ch_indexes, n_channels)[:, None, None, :]

    rgb_images = indices * images
    rgb_images += min_val * (1 - indices)

    return rgb_images


def search_file_tree(directory, filetypes=None, write=True):

    directory = Path(directory)
    if (directory / "file_paths.txt").is_file():
        log.info(
            "Using pregenerated txt file in the following directory for reading file paths: "
        )
        log.info(log.yellow(directory))
        with open(directory / "file_paths.txt", encoding="utf-8") as file:
            file_paths = file.read().splitlines()
        return file_paths

    if filetypes is None:
        filetypes = ("jpg", "jpeg", "JPEG", "png", "PNG")

    file_paths = []

    log.info(f"\nSearching file tree: {Path(directory)}\n")
    file_paths = []
    for ext in filetypes:
        file_paths.extend(list(Path(directory).rglob(f"*.{ext}")))
    assert len(file_paths) > 0, f"No image files were found in {directory}"

    if write:
        with open(directory / "file_paths.txt", "w", encoding="utf-8") as file:
            _file_paths = [str(file) + "\n" for file in file_paths]
            file.writelines(_file_paths)

    return file_paths


def postprocess_image(data, normalization_range=None, output_range=None):

    if normalization_range is None:
        normalization_range = ops.min(data), ops.max(data)

    if output_range is None:
        output_range = (0, 255)

    data = ops.convert_to_numpy(data)
    data = ops.clip(data, *normalization_range)
    data = translate(data, normalization_range, output_range)

    if ops.any(ops.isnan(data)):
        return ops.clip(data, output_range[0], output_range[1])

    data = ops.convert_to_numpy(data)
    return np.clip(data, output_range[0], output_range[1]).astype("uint8")


def get_postprocess_fn(config=None):
    if config.data.get("hdf5_key") == "data/complex_image":
        return lambda x: mri.complex_abs(x)

    return lambda x: postprocess_image(x, config.data.normalization)


def plot_batch(
    train_batch, val_batch, save_path=None, aspect=None, vmin=None, vmax=None
):
    num_images = train_batch.shape[0]
    nrows = int(math.sqrt(num_images))
    ncols = math.floor(num_images / nrows)

    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor("black")
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols * 2), axes_pad=0.0)
    grid = np.array(grid).reshape((nrows, ncols * 2))
    image_shape = train_batch[0].shape
    if aspect is None:
        aspect = image_shape[1] / image_shape[0]

    if image_shape[-1] == 1 or len(image_shape) == 2:
        cmap = "gray"
    else:
        cmap = None

    i = 0
    for row in range(nrows):
        for col in range(ncols):
            # training image
            ax = grid[row, col]
            image = train_batch[i]
            ax.imshow(image, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)
            ax.axis("off")

            # validation image
            ax = grid[row, col + ncols]
            image = val_batch[i]
            ax.imshow(image, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)
            ax.axis("off")
            i += 1

    ax = grid[0, 0]
    ax.set_title("Training", color="white", fontsize=12)
    ax = grid[0, ncols]
    ax.set_title("Validation", color="white", fontsize=12)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        log.success(f"Saved plot to {log.yellow(save_path)}")
    plt.close(fig)
    return fig


def plot_image_grid(
    images: List[np.ndarray],
    ncols: Optional[int] = None,
    cmap: Optional[Union[str, List[str]]] = "gray",
    vmin: Optional[Union[float, List[float]]] = None,
    vmax: Optional[Union[float, List[float]]] = None,
    titles: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
    aspect: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fig: Optional[plt.Figure] = None,
    fig_contents: Optional[List] = None,
    remove_axis: Optional[bool] = True,
    background_color: Optional[str] = "black",
    text_color: Optional[str] = "white",
    **kwargs,
) -> Tuple[plt.Figure, List]:

    if ncols is None:
        factors = [i for i in range(1, len(images) + 1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    images = [images[i] if len(images) > i else None for i in range(nrows * ncols)]

    aspect_ratio = images[0].shape[1] / images[0].shape[0]
    if figsize is None:
        figsize = (ncols * 2, nrows * 2 / aspect_ratio)

    assert (fig is None) == (
        fig_contents is None
    ), "Supply both fig and fig_contents or neither"

    if fig is None:
        fig = plt.figure(figsize=figsize, **kwargs)
        axes = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.1)
        if background_color:
            fig.patch.set_facecolor(background_color)
        fig.set_tight_layout({"pad": 0.1})
    else:
        axes = fig.axes[: len(images)]

    if isinstance(cmap, str):
        cmap = [cmap] * len(images)
    else:
        if cmap is None:
            cmap = [None] * len(images)
        assert len(cmap) == len(
            images
        ), f"cmap must be a string or list of strings of length {len(images)}"

    if isinstance(vmin, (int, float)):
        vmin = [vmin] * len(images)
    else:
        if vmin is None:
            vmin = [None] * len(images)
        assert len(vmin) == len(
            images
        ), f"vmin must be a float or list of floats of length {len(images)}"

    if isinstance(vmax, (int, float)):
        vmax = [vmax] * len(images)
    else:
        if vmax is None:
            vmax = [None] * len(images)
        assert len(vmax) == len(
            images
        ), f"vmax must be a float or list of floats of length {len(images)}"

    if fig_contents is None:
        fig_contents = [None for _ in range(len(images))]
    for i, ax in enumerate(axes):
        image = images[i]
        if fig_contents[i] is None:
            im = ax.imshow(
                image, cmap=cmap[i], vmin=vmin[i], vmax=vmax[i], aspect=aspect
            )
            fig_contents[i] = im
        else:
            fig_contents[i].set_data(image)
        if remove_axis:
            ax.axis("off")
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        if titles:
            ax.set_title(titles[i], color=text_color)

    if suptitle:
        fig.suptitle(suptitle, color=text_color)

    fig.set_tight_layout(False)
    return fig, fig_contents


def sample_images(
    diffusion_model,
    image_shape,
    num_steps,
    save_path,
    measurement_img=None,
    operator=None,
    n_frames=1,
    animate_diffusion_process=False,
    postprocess_func=None,
    gt_img=None,
    guidance_method="dps",
    guidance_kwargs=None,
    dpi=400,
    save_generated_images=True,
    save_measurement=True,
    measurement_postprocess_fn=None,
    seed=None,
):

    assert not (
        bool(measurement_img is None) and bool(image_shape is None)
    ), "Either image_shape or measurement_img must be provided"

    if measurement_img is not None:
        image_shape = keras.ops.shape(measurement_img)

    assert (
        len(image_shape) == 4
    ), "image_shape must be a tuple of (batch, height, width, channels)"

    if guidance_kwargs is None:
        guidance_kwargs = {}

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)

    if measurement_img is not None:
        generated_images_raw = diffusion_model.posterior_sampling(
            measurement_img,
            operator,
            diffusion_steps=num_steps,
            guidance_method=guidance_method,
            guidance_kwargs=guidance_kwargs,
            verbose=True,
        )
    else:
        if "StableDiffusion" in str(diffusion_model):
            generated_images_raw = diffusion_model.text_to_image(
                prompt="fish",
                batch_size=image_shape[0],
                num_steps=num_steps,
                seed=seed,
            )
        else:
            generated_images_raw = diffusion_model.generate(
                image_shape=image_shape,
                diffusion_steps=num_steps,
                verbose=True,
                seed=seed,
            )

    if save_generated_images is False:
        return generated_images_raw

    if postprocess_func is None:
        postprocess_func = lambda x: postprocess_image(
            x,
            diffusion_model.image_range,
        )

    def _reshape(x):

        x = keras.ops.reshape(x, (*image_shape[:-1], -1, n_frames))
        return keras.ops.transpose(x, (4, 0, 1, 2, 3))

    generated_images = _reshape(generated_images_raw)
    if measurement_img is not None and save_measurement:
        measurement_img = _reshape(measurement_img)
    if gt_img is not None:
        gt_img = _reshape(gt_img)

    generated_images = ops.convert_to_numpy(generated_images)

    def _save_plotted_image(images, index, _save_path, tag):
        fig, _ = plot_image_grid(postprocess_func(images), background_color="black")
        if n_frames > 1:
            _save_path = str(_save_path).replace(".png", f"_{tag}_frame_{index}.png")
        else:
            _save_path = str(_save_path).replace(".png", f"_{tag}.png")
        fig.savefig(
            _save_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi, transparent=True
        )
        plt.close(fig)
        log.success(f"Saved samples to -> {log.yellow(_save_path)}")

    for i in range(n_frames):
        _save_plotted_image(generated_images[i], i, save_path, "generated")
        if measurement_img is not None and save_measurement:
            if measurement_postprocess_fn:
                measurement_img_to_save = measurement_postprocess_fn(measurement_img[i])
            else:
                measurement_img_to_save = measurement_img[i]
            _save_plotted_image(measurement_img_to_save, i, save_path, "measurement")
        if gt_img is not None:
            _save_plotted_image(gt_img[i], i, save_path, "gt")

    return generated_images_raw


def get_checkpoint_index(path, model_name=None):

    try:
        if model_name is None:
            path = int(re.findall(r"_(\d+)\.weights\.h5", str(path))[0])
        else:
            path = int(
                re.findall(r"{}.*_(\d+)\.weights\.h5".format(model_name), str(path))[0]
            )
    except IndexError as exc:
        match_str = "_<index>.weights.h5"
        if model_name is not None:
            match_str = f"{model_name}{match_str}*"
        raise ValueError(
            f"Could not extract checkpoint index from {path}, "
            f"does not match '{match_str}' format"
        ) from exc
    return path


def get_latest_checkpoint_path(directory, model_name=None):

    if model_name is None:
        all_checkpoint_paths = list(Path(directory).glob("*.weights.h5"))
    else:
        all_checkpoint_paths = list(Path(directory).glob(f"{model_name}*.weights.h5"))
    assert (
        len(all_checkpoint_paths) != 0
    ), f"Did not find any checkpoint files in {directory}"

    _get_checkpoint_index = lambda path: get_checkpoint_index(path, model_name)

    latest_checkpoint_path = max(
        all_checkpoint_paths, key=_get_checkpoint_index, default=None
    )
    checkpoint_path = str(latest_checkpoint_path)
    return checkpoint_path


def cumprod_to_orig(cumprod_array):

    cumprod_array = ops.convert_to_numpy(cumprod_array)
    assert np.all(cumprod_array > 0), "cumprod_array must be strictly positive"
    retrieved_array = np.zeros_like(cumprod_array)
    retrieved_array[0] = cumprod_array[0]
    retrieved_array[1:] = cumprod_array[1:] / cumprod_array[:-1]
    retrieved_array = ops.convert_to_tensor(retrieved_array)
    return retrieved_array


def get_size_in_bytes(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_ram_info():
    svmem = psutil.virtual_memory()
    data = {
        "total": get_size_in_bytes(svmem.total),
        "available": get_size_in_bytes(svmem.available),
        "used": get_size_in_bytes(svmem.used),
        "percentage": str(f"{svmem.percent}"),
    }
    return data


def load_img_as_tensor(path, image_shape=None, grayscale=False):

    assert Path(path).is_file(), f"File {path} does not exist."
    assert len(image_shape) == 2, "image_shape must be a tuple of (height, width)"
    image = Image.open(path)
    if grayscale:
        if image.mode != "L":
            image = image.convert("L")
    else:
        if image.mode == "L":
            image = image.convert("RGB")

    if grayscale:
        image = np.expand_dims(image, axis=-1)

    if image_shape is not None:
        image = ops.convert_to_tensor(image)
        image = ops.image.resize(image, size=image_shape, antialias=True)
    image = np.array(image, dtype=np.float32)
    if image.shape[2] == 4:
        image = image[:, :, :3]  # remove alpha channel
    return ops.convert_to_tensor(image)


def normalize(x):
    """
    Normalizes x to range [-1, 1]
    """
    x = x - ops.min(x)
    x = x / ops.max(x)
    x = (x - 0.5) / 0.5
    return x
