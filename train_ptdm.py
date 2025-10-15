"""
online fine-tuning h-model
"""

import argparse
import os
import sys
from pathlib import Path


import cv2
import numpy as np
import keras
import tensorflow as tf
import json
import logging
from pathlib import Path
from tensorflow import keras

import utils.lib.utils as lib_utils
import yaml
from callbacks import DDIMSamplingCallback
from keras import ops
from utils.lib.config import Config, load_config_from_yaml

from datasets import get_datasets
from load_model import load_model
from ddim import DiffusionModel

from utils.keras_utils import ClearMemory, get_loss_func, get_postprocess_fn, plot_batch
keras.utils.get_custom_objects()["DiffusionModel"] = DiffusionModel
from tensorflow.keras.callbacks import Callback

class HModelCheckpoint(Callback):
    def __init__(self, filepath, save_best_only=False, monitor="val_loss", mode="min"):
        super().__init__()
        self.filepath = filepath  
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            print(f"⚠️ Metric '{self.monitor}' not found in logs. Skipping saving.")
            return

        if self.best is None:
            self.best = current

        should_save = True
        if self.save_best_only:
            if self.mode == "min":
                should_save = current < self.best
            elif self.mode == "max":
                should_save = current > self.best
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

        if should_save:
            formatted_path = self.filepath.format(epoch=epoch + 1)
            print(f"💾 Saving h_trans_model weights to {formatted_path} ...")
            self.model.h_trans_model.save_weights(formatted_path)
            self.best = current


def print_train_summary(config):
    """Print training summary from config."""
    print("=" * 57)
    print("Training Summary:")
    print("=" * 57)
    print(f"| {'Parameter':<20} | {'Value':<30} |")
    print("|" + "-" * 55 + "|")
    print(f"| {'Epochs':<20} | {config.optimization.num_epochs:<30} |")
    print(f"| {'Learning rate':<20} | {config.optimization.learning_rate:<30} |")
    print(
        f"| {'Latent diffusion':<20} | {'ON' if config.model.latent else 'OFF':<30} |"
    )
    print(
        f"| {'Image shape':<20} | {', '.join(str(dim) for dim in config.data.image_shape_after_augmentations):<30} |"
    )
    if config.model.latent:
        if config.model.get("latent_shape") is not None:
            print(
                f"| {'Latent shape':<20} | "
                f"{', '.join(str(dim) for dim in config.model.latent_shape):<30} |"
            )
    print(
        f"| {'Normalized range':<20} | "
        f"{', '.join(str(val) for val in config.data.normalization):<30} |"
    )
    print(f"| {'#Frames':<20} | {config.data.n_frames:<30} |")
    if config.data.extension == "tf_dataset":
        dataset = f"{config.data.dataset_folder}/{config.data.dataset_name}/{config.data.dataset_version}"
    else:
        dataset = config.data.train_folder

    if isinstance(dataset, list):
        dataset = [Path(folder).name for folder in dataset]
        dataset = ",".join(dataset)
    else:
        if config.data.extension == "tf_dataset":
            dataset = ":".join(dataset.split("/")[-2:])
        else:
            dataset = Path(dataset).name
    print(f"| {'Dataset':<20} | {dataset:<30} |")
    print("=" * 57)


def train_ptdm(
    diffusion_model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    config: Config,
    run_dir: str,
    postprocess_func: callable,
    train: bool = True,
) -> DiffusionModel:
    

    from h_trans import HTransformModel
    weight_decay = config.optimization.weight_decay
    learning_rate = config.optimization.learning_rate
    num_epochs = config.optimization.num_epochs
    widths = config.model.widths
    block_depth = config.model.block_depth



    widths_hmodel = config.model.widths
    block_depth_hmodel = config.model.block_depth


    ema_val = config.optimization.ema
    min_signal_rate = config.sampling.min_signal_rate
    max_signal_rate = config.sampling.max_signal_rate
    diffusion_steps = config.model.diffusion_steps
    run_eagerly = config.model.run_eagerly
    latent = config.model.latent

    image_shape = train_dataset.element_spec[1].shape[1:].as_list()
    print(f"ddim shape{train_dataset.element_spec}")

    if config.data.get("image_shape_after_augmentations") is None:
        config.data.image_shape_after_augmentations = config.data.image_shape

    # if specified in config, check if image shape from dataset matches the specified image shape
    specified_image_shape = config.data.image_shape_after_augmentations
    print(f"shape: {specified_image_shape, specified_image_shape[:-1], specified_image_shape[-1]}")
    if specified_image_shape:

        assert image_shape == specified_image_shape[:-1] + [
            specified_image_shape[-1] * config.data.n_frames
        ], (
            f"Image shape from dataset {image_shape} does not match image shape "
            f"specified in config {specified_image_shape} with n_frames {config.data.n_frames}."
        )
    else:
        config.data.image_shape_after_augmentations = image_shape[:-1] + [
            image_shape[-1] // config.data.n_frames
        ]

    run_eagerly = config.model.run_eagerly if not sys.gettrace() else True

    
    h_model = DiffusionModel(
        image_shape,
        widths,
        block_depth,
        ema_val=ema_val,
        min_signal_rate=min_signal_rate,
        max_signal_rate=max_signal_rate,
        diffusion_steps=diffusion_steps,
        image_range=config.data.normalization,
        latent_diffusion=latent,
        latent_shape=config.model.latent_shape if latent else None,
        autoencoder_checkpoint_directory=config.model.get(
            "autoencoder_checkpoint_directory"
        ),
    )
    
    
    
    model = HTransformModel(
        image_shape,
        diffusion_model,
        h_model,
    )


    checkpoint_path = Path(run_dir) / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)

    model.h_trans_model.save_model_json(checkpoint_path)

    loss_func = get_loss_func(config.optimization.loss)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=loss_func,
        run_eagerly=run_eagerly,
    )  


    checkpoint_file = str(checkpoint_path / "diffusion_model_{epoch}.weights.h5")

    h_model_ckpt_callback = HModelCheckpoint(
        filepath=checkpoint_file,
        monitor="i_loss",
        mode="min",
        save_best_only=False,
    )

    validation_sample_callback = DDIMSamplingCallback(
        model,
        image_shape,
        config.evaluation.diffusion_steps,
        config.evaluation.batch_size,
        save_dir=run_dir / "samples",
        n_frames=config.data.n_frames,
        postprocess_func=postprocess_func,
        start_with_eval=config.evaluation.get("start_with_eval", False),
    )

    config.save_to_yaml(Path(run_dir) / "config.yaml")

    print_train_summary(config)

    callbacks = [
        validation_sample_callback,
        #checkpoint_callback,
        h_model_ckpt_callback,
        ClearMemory(),
    ]

    if train:



        start_training_str = (
            f"Starting training for {num_epochs} "
            f"epochs on {lib_utils.get_date_string()}..."
        )
        print("-" * len(start_training_str))
        print(start_training_str)
        print(f"model is built: {model.built}")
        print(hasattr(model, "call"))
        from keras import backend
        print("Current Keras backend:", backend.backend())

        model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=None,#val_dataset,
            callbacks=callbacks,
            steps_per_epoch=config.optimization.get("steps_per_epoch"),
        )

    h_model = model.h_trans_model

    return h_model


def parse_args():
    parser = argparse.ArgumentParser(description="DDIM training")
    parser.add_argument(
        "-c",
        "--train_config",
        type=str,
        default="configs/training/ddim_train_fastmri.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-d",
        "--data_root",
        type=str,
        default="data/",
        help="Path to the your data directory.",
    )
    parser.add_argument(
        "-m",
        "--mask_path",
        type=str,
        default="temp_tst/",
        help="Path to the your data directory.",
    )
    parser.add_argument(
        "-o",
        "--image_path",
        type=str,
        default="temp_tst/",
        help="Path to the your data directory.",
    )

    return parser.parse_args()


def generate_mask(image_shape, mask_ratio=0.9):

    h, w = image_shape
    total_pixels = h * w
    mask_pixels = int(total_pixels * mask_ratio)

    mask = np.ones((h, w), dtype=np.uint8)
    mask_flat = mask.flatten()
    zero_indices = np.random.choice(total_pixels, mask_pixels, replace=False)
    mask_flat[zero_indices] = 0
    mask = mask_flat.reshape((h, w))

    print(mask)

    return mask


if __name__ == "__main__":
    # load config from yaml file
    args = parse_args()
    train_config = load_config_from_yaml(Path(args.train_config), loader=yaml.UnsafeLoader)
    data_root = args.data_root
    train_config.data.__setattr__("user", {"data_root": data_root})



    batch_size = 512

    img = cv2.imread(Path(args.image_path), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))


    img = img.astype('float32') / 127.5 - 1.0
    image_shape = (img.shape[0], img.shape[1])

    batch_images = np.repeat(img[np.newaxis, :, :, :], batch_size, axis=0)

    

    mask = cv2.imread(Path(args.mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128))
    mask = mask.astype('float32') / 255

    channels = img.shape[-1]

    mask = np.repeat(mask[..., np.newaxis], channels, axis=-1)

    batch_mask = np.repeat(mask[np.newaxis, :, :], batch_size, axis=0)

    masked_images = batch_mask*batch_images
    

    first_masked_image = masked_images[0]

    first_masked_image_uint8 = ((first_masked_image+1) * 127.5).astype(np.uint8)



    train_data = tf.data.Dataset.from_tensor_slices((batch_mask, masked_images)).batch(64)

    image_shape = train_data.element_spec[1].shape[1:].as_list()


    h_model_run_dir = Path(str(train_config.h_model_run_dir)) 
    h_model_run_dir.mkdir(exist_ok=True, parents=True)



    postprocess_func = get_postprocess_fn(train_config)
    print(f"loading from {train_config}")

    diffusion_model = load_model(
            Path(str(train_config.diffusion_model_run_dir)) / "checkpoints",
            stable_diffusion_kwargs={
                "img_height": image_shape[0],
                "img_width": image_shape[1],
            },
            image_shape=image_shape,
        )

    

    print(train_data)

    model = train_ptdm(
        diffusion_model,
        train_data,
        None,
        train_config,
        h_model_run_dir,
        postprocess_func,
        train=True,
    )
