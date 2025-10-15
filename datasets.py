import itertools
import re
from functools import partial
from pathlib import Path
from typing import Optional

import h5py
import albumentations as A
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RandomCrop, Resizing, ZeroPadding2D
import tensorflow_datasets as tfds
from utils.lib import log
from utils.lib.config import Config

from utils.keras_utils import (
    get_normalization_layer,
    grayscale_to_random_rgb,
    search_file_tree,
    translate,
)


def read_image(image_path, channels=3):
    """Read image from file path."""
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels)

    return image


def read_numpy(numpy_path):
    np_array = np.load(numpy_path)
    return tf.convert_to_tensor(np_array, dtype=tf.float32)


def preprocess_image(image, image_shape):
    """Prepocess image, resize and normalize."""
    # image = tf.image.random_crop(image, size=[*image_shape, 1])

    image = tf.image.resize(image, size=image_shape, antialias=True)
    return image


def prepare_image_dataset(
    dataset_folder: str,
    image_shape: tuple,
    batch_size: int,
    augmenter: keras.Sequential,
    dataset_repetitions: int = 1,
    num_img: Optional[int] = None,
    normalization_range: Optional[tuple] = None,
    image_range: Optional[tuple] = None,
    shuffle: Optional[bool] = True,
):
    """Prepare image dataset for training.

    Does the following in order:
    - Search for images in the dataset folder.
    - Read images from file.
    - Preprocess images (resize and normalize).
    - Cache images.
    - Repeat dataset (if dataset_repetitions > 1).
    - Shuffle dataset.
    - Batch dataset.
    - Augment images.
    - Prefetch dataset.

    Args:
        dataset_folder (str): Path to the dataset folder.
        image_shape (tuple): Shape of the images in the dataset.
        batch_size (int): Batch size for training.
        augmenter (keras.Sequential): Image augmentation model.
        dataset_repetitions (int, optional): Number of times to repeat the dataset.
            Defaults to 1.
        num_img (Optional[int], optional): Number of images to use from the dataset.
            Defaults to None. If None, all images are used.

    Returns:
        tf.data.Dataset: Prepared image dataset for training.
    """
    image_paths = search_file_tree(dataset_folder)
    num_img_dataset = len(image_paths)
    if num_img_dataset == 0:
        raise ValueError(
            f"Dataset folder {dataset_folder} does not contain any images."
        )
    else:
        print(f"Found {num_img_dataset} images in {dataset_folder}")
    image_paths = [str(path) for path in image_paths]
    if num_img is not None:
        # select random subset of images
        image_paths = np.random.choice(image_paths, num_img, replace=False)
        print(
            f"Using {num_img} images ({(num_img / num_img_dataset) * 100:.2f}%) "
            f"from {dataset_folder}"
        )

    assert tuple(image_range) == (
        0,
        255,
    ), f"Image datasets must have range (0, 255), got {image_range}"

    if normalization_range is not None:
        normalizer = get_normalization_layer(*normalization_range, *image_range)
    else:
        normalizer = keras.layers.Lambda(lambda x: x)

    dataset = (
        tf.data.Dataset.from_tensor_slices(image_paths)
        .cache()
        .repeat(dataset_repetitions)
        .map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
        .map(
            lambda x: preprocess_image(x, image_shape),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(normalizer, num_parallel_calls=tf.data.AUTOTUNE)
    )
    if shuffle:
        dataset = dataset.shuffle(10 * batch_size)

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    print(f"prepare dataset: {dataset}")

    if augmenter is not None:
        dataset = dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def prepare_tf_dataset(
    dataset_name: str,
    dataset_version: str,
    dataset_folder: str,
    split: str,
    batch_size: int,
    image_shape: tuple,
    augmenter: keras.Sequential,
    dataset_repetitions: int = 1,
    normalization_range: Optional[tuple] = None,
    image_range: Optional[tuple] = None,
    shuffle: Optional[bool] = True,
):
    builder = tfds.builder(
        dataset_name,
        version=dataset_version,
        data_dir=dataset_folder,
    )
    try:
        dataset = builder.as_dataset(
            split=split,
            shuffle_files=shuffle,
            batch_size=batch_size,
            as_supervised=False,
        )
    except Exception as exc:
        log.warning("Failed to load dataset, trying to download it...")
        error_msg = f"Failed to download and prepare dataset {dataset_name}:{dataset_version} from {dataset_folder}"
        try:
            builder.download_and_prepare()
            dataset = builder.as_dataset(
                split=split,
                shuffle_files=shuffle,
                batch_size=batch_size,
                as_supervised=False,
            )
        except Exception as exc:
            raise ValueError(error_msg + f"\n{exc}") from exc

        raise ValueError(error_msg + f"\n{exc}") from exc

    assert tuple(image_range) == (
        0,
        255,
    ), f"Image datasets must have range (0, 255), got {image_range}"

    if normalization_range is not None:
        normalizer = get_normalization_layer(*normalization_range, *image_range)
    else:
        normalizer = keras.layers.Lambda(lambda x: x)

    if image_shape is not None:
        resize_layer = keras.layers.Resizing(*image_shape)
    else:
        resize_layer = keras.layers.Lambda(lambda x: x)

    dataset = (
        dataset.cache()
        # extract image from each batch
        .map(lambda x: x["image"], num_parallel_calls=tf.data.AUTOTUNE)
        .repeat(dataset_repetitions)
        .map(resize_layer, num_parallel_calls=tf.data.AUTOTUNE)
        .map(normalizer, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    if augmenter is not None:
        dataset = dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


class H5Generator:
    """Generator from h5 file."""

    def __init__(self, n_frames=1, frames_dim=-1, new_frames_dim=False):
        """Initialize H5Generator.
        Args:
            n_frames (int, optional): number of frames to load from file.
                Defaults to 1. Frames are read in sequence and stacked along
                the last axis (channel).
            frames_dim (int, optional): dimension to stack frames along.
                Defaults to -1. If new_frames_dim is True, this will be the
                new dimension to stack frames along.
            new_frames_dim (bool, optional): if True, new dimension to stack
                frames along will be created. Defaults to False. In that case
                frames will be stacked along existing dimension (frames_dim).
        """
        self.file_name = None
        self.n_frames = n_frames
        self.frames_dim = frames_dim
        self.new_frames_dim = new_frames_dim

    def __call__(self, file_name, key):
        """Yields image from h5 file using key.

        Args:
            file_name (str): file name of h5 file.
            key (str): key of dataset in h5 file.

        Yields:
            np.ndarray: image of shape [n_frames, shape].
                where shape is image_shape + (n_channels * n_frames,).
        """
        with h5py.File(file_name, "r") as file:
            n_frames_tot = len(file[key])
            for i in range(n_frames_tot - (self.n_frames - 1)):
                images = []
                for j in range(self.n_frames):
                    image = file[key][i + j]
                    images.append(image)
                if self.new_frames_dim:
                    images = np.stack(images, axis=self.frames_dim)
                else:
                    images = np.concatenate(images, axis=self.frames_dim)

                self.file_name = Path(str(file_name)).stem + "_" + str(i)
                yield images

    def length(self, file_name, key):
        """Return length (number of elements) in h5 file.
        Args:
            file_name (str): file name of h5 file.
            key (str): key of dataset in h5 file.
        Returns:
            int: length of dataset (number of frames).
                if n_frames > 1, the length will be reduced by n_frames - 1.
                this is because when indexing always n_frames are read at the
                same time, effectively reducing the length of the dataset.
        """
        with h5py.File(file_name, "r") as file:
            return len(file[key]) - (self.n_frames - 1)



def get_datasets(
    data_root: str,
    config: Config,
    batch_size=None,
    shard_index: int = 0,
    num_shards: int = 0,
    **kwargs,
):
    """Get training and validation datasets.

    Args:
        data_root (str): Path to the data root.
        config (Config): The configuration object.
        augmenter (Optional[keras.Sequential], optional): Image augmentation model.
            Defaults to None.
    Returns:
        tf.data.Dataset: Training dataset.
        tf.data.Dataset: Validation dataset.
    """
    data_root = Path(data_root)

    if config.data.image_shape:
        image_shape = config.data.image_shape[:2]
    else:
        image_shape = None
    print(f"dataset: {image_shape, config.data.image_shape[:2]}")

    shuffle = config.data.get("shuffle", True)
    if not shuffle:
        log.warning("shuffle is set to False, dataset will not be shuffled")

    if config.data.extension != "tf_dataset":
        if config.data.train_folder == config.data.val_folder:
            log.warning(
                f"train_folder and val_folder are the same: {config.data.train_folder}"
            )

    if config.data.get("image_shape_after_augmentations") is None:
        config.data.image_shape_after_augmentations = config.data.image_shape

    ## augmentation
    train_augmenter = None
    val_augmenter = None
    if "augmentation" in config.data and config.data.augmentation is not None:
        assert isinstance(
            config.data.augmentation, dict
        ), "config.data.augmentation must be a dictionary"
        assert set(config.data.augmentation.keys()) == {
            "training",
            "validation",
        }, "config.data.augmentation must have keys 'training' and 'validation'"

        train_augmenter = get_augmentation(
            config.data.augmentation["training"],
            config.data.image_shape_after_augmentations,
        )
        val_augmenter = get_augmentation(
            config.data.augmentation["validation"],
            config.data.image_shape_after_augmentations,
        )

    if config.data.extension in ("jpg", "jpeg", "JPEG", "png", "PNG"):
        assert (
            num_shards == 0
        ), "parallel not supported for jpg, jpeg, and png dataloader"
        assert (
            shard_index == 0
        ), "num_parallel not supported for jpg, jpeg, and png dataloader"
        log.warning("jpg, jpeg, and png image dataloader not supported.")
        assert isinstance(config.data.train_folder, str), (
            "train_folder must be a string for image dataset dataloader, "
            f"got {config.data.train_folder}"
        )
        if config.data.n_frames != 1:
            raise ValueError(
                "n_frames must be 1 for jpg, jpeg, or png image dataloader, "
                f"got {config.data.n_frames}"
            )
        #print(image_shape)
        #image_shape = [250,250,3]
        train_dataset = prepare_image_dataset(
            data_root / config.data.train_folder,
            image_shape,
            batch_size or config.data.batch_size,
            train_augmenter,
            dataset_repetitions=config.data.dataset_repetitions,
            num_img=config.data.num_img,
            image_range=config.data.range,
            normalization_range=config.data.normalization,
            **kwargs,
        )

        val_dataset = prepare_image_dataset(
            data_root / config.data.val_folder,
            image_shape,
            batch_size or config.data.batch_size,
            val_augmenter,
            dataset_repetitions=config.data.dataset_repetitions,
            num_img=config.data.num_img,
            image_range=config.data.range,
            normalization_range=config.data.normalization,
            **kwargs,
        )
    elif config.data.extension == "tf_dataset":
        train_dataset = prepare_tf_dataset(
            config.data.dataset_name,
            config.data.dataset_version,
            config.data.dataset_folder,
            "train",
            batch_size or config.data.batch_size,
            image_shape,
            train_augmenter,
            dataset_repetitions=config.data.dataset_repetitions,
            image_range=config.data.range,
            normalization_range=config.data.normalization,
            shuffle=shuffle,
        )
        val_dataset = prepare_tf_dataset(
            config.data.dataset_name,
            config.data.dataset_version,
            config.data.dataset_folder,
            "test",
            batch_size or config.data.batch_size,
            image_shape,
            val_augmenter,
            dataset_repetitions=config.data.dataset_repetitions,
            image_range=config.data.range,
            normalization_range=config.data.normalization,
            shuffle=shuffle,
        )
    else:
        raise ValueError(
            f"Unsupported data extension {config.data.extension}"
            f"Please choose from jpg, jpeg, png, hdf5, h5."
        )
    return train_dataset, val_dataset


class Sequential:
    def __init__(self, image_shape_after_augmentations):
        self.layers = []
        self.image_shape_after_augmentations = image_shape_after_augmentations

    def add(self, layer):
        self.layers.append(layer)

    def add_albumenation(self, layer):
        """albumentations"""
        self.layers.append(lambda x: self.batched(x, layer))

    def aug_fn(self, image, albumentation):
        """albumentations"""
        return albumentation(image=image)["image"]

    def process_data(self, image, albumentation):
        """albumentations"""
        return tf.numpy_function(
            func=partial(self.aug_fn, albumentation=albumentation),
            inp=[image],
            Tout=tf.float32,
        )

    def batched(self, image, albumentation):
        """albumentations"""
        return tf.map_fn(
            partial(self.process_data, albumentation=albumentation),
            image,
        )

    def __len__(self):
        return len(self.layers)

    def __call__(self, x):
        # keras layers
        for layer in self.layers:
            x = layer(x)
        # TODO: make this work properly
        try:
            x.set_shape([None, *self.image_shape_after_augmentations])
        except:
            pass
        return x


def get_augmentation(augmentation_config: list, image_shape_after_augmentations):
    """Get image augmentation model based on the configuration.

    Args:
        augmentation_config (list): Configuration for image augmentation.
            example: ["random_flip"]
        image_shape_after_augmentations (list): Shape of the images after augmentation.
            [height, width, channels]

    Returns:
        callable: Image augmentation model.
    """
    # When augmentation_config is None, return None
    if augmentation_config is None:
        return None

    # Merge list of strings and dicts into a single dict
    augmentation_config_ = {}
    for item in augmentation_config:
        if isinstance(item, dict):
            augmentation_config_.update(item)
        elif isinstance(item, str):
            augmentation_config_[item] = {}
    aug_config = augmentation_config_

    # Assert that all keys in reformatted_augmentation_config are valid
    augmentation_types = aug_config.keys()
    all_augmentation_types = {
        "random_flip",
        "random_resized_crop",
        "resize",
        "random_rgb",
    }
    all_augmentation_types_ = ", '".join(all_augmentation_types)
    assert (
        set(augmentation_types) <= all_augmentation_types
    ), f"augmentation_config must have keys in '{all_augmentation_types_}'"

    # Build the augmentation model
    layers = Sequential(image_shape_after_augmentations)
    for augmentation_type in augmentation_types:
        if augmentation_type == "random_flip":
            layers.add(keras.layers.RandomFlip(**aug_config["random_flip"]))
        elif augmentation_type == "random_resized_crop":
            layers.add_albumenation(
                A.RandomResizedCrop(**aug_config["random_resized_crop"])
            )
        elif augmentation_type == "resize":
            layers.add_albumenation(A.Resize(**aug_config["resize"]))
        elif augmentation_type == "random_rgb":
            layers.add(keras.layers.Lambda(grayscale_to_random_rgb))

    return layers
