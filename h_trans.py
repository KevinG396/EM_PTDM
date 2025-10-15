
from pathlib import Path
import os

import h5py
import keras
import subprocess
import tensorflow as tf
from keras import ops
from utils.lib import log
import numpy as np
from guidance.latent_guidance import get_guidance as get_latent_guidance
from guidance.pixel_guidance import get_guidance as get_pixel_guidance
from load_model import load_model
from models.unet import get_network
from utils.keras_utils import get_ram_info, get_loss_func, get_postprocess_fn
from PIL import Image
from ddim import DiffusionModel
#from train_ddeft import train_deft
from train_ptdm import train_ptdm
def get_xt_condition(xt):

    xt_condition = xt
    return xt_condition

def save_image_grid(images, save_path, grid_size=(4, 4)):

    B, H, W = images.shape[:3]
    assert B == grid_size[0] * grid_size[1], f"img num {B} doesn't match {grid_size}"

    if images.ndim == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)  # (B, H, W)
    
    if images.ndim == 3:
        canvas = np.zeros((H * grid_size[0], W * grid_size[1]), dtype=np.uint8)
        mode = "L"
    elif images.ndim == 4 and images.shape[-1] == 3:
        canvas = np.zeros((H * grid_size[0], W * grid_size[1], 3), dtype=np.uint8)
        mode = "RGB"
    else:
        raise ValueError(f"unsupport dim: {images.shape}")

    for idx, image in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        if mode == "L":
            canvas[row * H:(row + 1) * H, col * W:(col + 1) * W] = image
        else:  # RGB
            canvas[row * H:(row + 1) * H, col * W:(col + 1) * W, :] = image

    Image.fromarray(canvas, mode=mode).save(save_path)
    print(f"Saved: {save_path}")

def save_single_image(image, save_path):

    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)  # (H, W)

    if image.ndim == 2:
        mode = "L"
    elif image.ndim == 3 and image.shape[-1] == 3:
        mode = "RGB"
    else:
        raise ValueError(f"unsupport dim: {image.shape}")

    Image.fromarray(image.astype(np.uint8), mode=mode).save(save_path)
    print(f"Saved: {save_path}")

@keras.saving.register_keras_serializable()
class HTransformModel(keras.Model):
    def __init__(
        self,
        image_shape,
        diffusion_model,
        h_trans_model,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_shape = image_shape


        self.diffusion_model = diffusion_model
        self.diffusion_model.trainable = False

        self.h_trans_model = h_trans_model


        self.ema_val = self.h_trans_model.ema_val
        self.latent_shape = self.h_trans_model.latent_shape
        self.min_signal_rate = self.h_trans_model.min_signal_rate
        self.max_signal_rate = self.h_trans_model.max_signal_rate
        self.diffusion_steps = self.h_trans_model.diffusion_steps
        self.image_range = self.h_trans_model.image_range

        self.mean = self.h_trans_model.mean
        self.variance = self.h_trans_model.variance
        self.latent_diffusion = self.h_trans_model.latent_diffusion
        self.autoencoder_checkpoint_directory = self.h_trans_model.autoencoder_checkpoint_directory

        self._image_encoder = None
        self._decoder = None
        self.noise_loss_tracker = None
        self.image_loss_tracker = None

        self.img_height, self.img_width = self.image_shape[:2]

        self.h_network = self.h_trans_model.network

        self.ema_h_network = self.h_trans_model.ema_network 

        self.track_progress = []

        assert len(self.image_range) == 2, "image_range must be a tuple of (min, max)"
        assert self.image_range[0] < self.image_range[1], "min must be less than max"
        assert self.min_signal_rate < self.max_signal_rate, "min must be less than max"

        if self.latent_diffusion:
            if self.autoencoder_checkpoint_directory is None:
                raise ValueError(
                    "latent_diffusion requires a pretrained autoencoder model to be specified"
                )
            self.autoencoder = load_model(
                self.autoencoder_checkpoint_directory,
                image_shape=image_shape,
                old_backbone=old_vae_backbone,
            )
            assert self.autoencoder.image_range == self.image_range, (
                f"image_range mismatch between autoencoder ({self.autoencoder.image_range}) "
                f"and DiffusionModel ({self.image_range})"
            )

            if self.autoencoder.latent_dim is not None:
                if hasattr(self.autoencoder, "latent_shape"):
                    assert self.autoencoder.latent_shape == self.latent_shape, (
                        f"latent_shape mismatch between autoencoder ({self.autoencoder.latent_shape}) "
                        f"and DiffusionModel ({self.latent_shape}). Please update this in your configs."
                    )
        else:
            assert (
                self.autoencoder_checkpoint_directory is None
            ), "latent_diffusion must be True if autoencoder_checkpoint_directory is specified"

    def compile(self, run_eagerly=None, jit_compile="auto", **kwargs):
        super().compile(run_eagerly=run_eagerly, jit_compile=jit_compile, **kwargs)


        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        if self.latent_diffusion:
            self.autoencoder.compile(run_eagerly=run_eagerly, jit_compile=jit_compile)

        if jit_compile:
            log.info("Model has been JIT compiled")
        if run_eagerly:
            log.warning("Model is running eagerly")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def diffusion_schedule(self, diffusion_times):

        # diffusion times -> angles
        start_angle = ops.cast(ops.arccos(self.max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(self.min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):

        if training:
            h_network = self.h_trans_model.network
        else:
            h_network = self.h_trans_model.ema_network

        pred_noises_eps1, pred_images_0 = self.diffusion_model.denoise(noisy_images, noise_rates, signal_rates, training=False)

        xt_condition = get_xt_condition(
                    xt=noisy_images

                )

        pred_noises_eps2 = h_network([xt_condition, noise_rates**2], training=training)

        pred_noises = pred_noises_eps1 + pred_noises_eps2

        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps,
        initial_samples=None,
        initial_step=0,
        verbose=False,
    ):
        num_images = ops.shape(initial_noise)[0]
        step_size = 1.0 / ops.cast(diffusion_steps, "float32")
        print("start revers step\n")


        progbar = keras.utils.Progbar(diffusion_steps, verbose=verbose)

        assert (
            initial_step >= 0
        ), f"initial_step must be non-negative, got {initial_step}"
        assert (
            initial_step < diffusion_steps
        ), f"initial_step must be less than diffusion_steps, got {initial_step}"

        if initial_samples is not None and initial_step > 0:
            starting_diffusion_times = ops.ones((num_images, 1, 1, 1)) - (
                (initial_step - 1) * step_size
            )
            noise_rates, signal_rates = self.diffusion_schedule(
                starting_diffusion_times
            )
            next_noisy_images = (
                signal_rates * initial_samples + noise_rates * initial_noise
            )
        else:
            next_noisy_images = initial_noise


        self.track_progress = []
        if diffusion_steps > 50:
            track_progress_interval = diffusion_steps // 50
        else:
            track_progress_interval = 1

        for step in range(initial_step, diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            progbar.update(step + 1)

            if step % track_progress_interval == 0:
                self.track_progress.append(pred_images)

        return pred_images

    def get_guidance_fn(
        self,
        guidance_method,
        operator,
    ):

        if self.latent_diffusion:
            guidance_fn = get_latent_guidance(guidance_method)(self)
        else:
            guidance_fn = get_pixel_guidance(guidance_method)(self)

        def run_guidance(
            noisy_images,
            measurement,
            noise_rates,
            signal_rates,
            next_signal_rates,
            next_noise_rates,
            **guidance_kwargs,
        ):

            gradients, (measurement_error, (pred_noises, pred_images)) = guidance_fn(
                noisy_images,
                measurement=measurement,
                operator=operator,
                noise_rates=noise_rates,
                signal_rates=signal_rates,
                **guidance_kwargs,
            )
            gradients = ops.nan_to_num(gradients)
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            next_noisy_images = next_noisy_images - gradients
            return next_noisy_images, measurement_error, pred_images

        return run_guidance

    def generate(
        self,
        image_shape,
        diffusion_steps=None,
        initial_samples=None,
        initial_step=0,
        verbose=False,
        seed=None,
    ):
        if diffusion_steps is None:
            diffusion_steps = self.diffusion_steps
        assert (
            len(image_shape) == 4
        ), "image_shape must be a tuple of (batch, height, width, channels)"
        num_images, image_height, image_width, n_channels = image_shape

        if self.latent_diffusion:
            if self.latent_shape is None:
                image_height = image_height // 8
                image_width = image_width // 8
                n_channels = 4
            else:
                image_height, image_width, n_channels = self.latent_shape

        initial_noise = keras.random.normal(
            shape=(num_images, image_height, image_width, n_channels),
            seed=seed,
        )
        if verbose:
            print("Generating images...")
        generated_images = self.reverse_diffusion(
            initial_noise,
            diffusion_steps,
            initial_samples=initial_samples,
            initial_step=initial_step,
            verbose=verbose,
        )

        if self.latent_diffusion:
            generated_images = self.decoder(generated_images)

        return generated_images


    def train_step(self, data):


        patch_mask_batch, masked_images = data


        patch_mask_batch = tf.cast(patch_mask_batch, tf.float32)

        if self.latent_diffusion:
            masked_images = self.image_encoder(masked_images)
        print(masked_images.shape)
        batch_size, image_height, image_width, n_channels = ops.shape(masked_images)


        noises = keras.random.normal(
            shape=(batch_size, image_height, image_width, n_channels)
        )

        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )


        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * masked_images + noise_rates * noises

        pred_noises_eps1, pred_images_0 = self.diffusion_model.denoise(noisy_images, noise_rates, signal_rates, training=False)

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(patch_mask_batch*(noises-pred_noises_eps1), patch_mask_batch*(pred_noises-pred_noises_eps1))  # used for training
            image_loss = self.loss(patch_mask_batch*masked_images, patch_mask_batch*pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.h_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.h_network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        for weight, ema_weight in zip(self.h_network.weights, self.ema_h_network.weights):
            ema_weight.assign(self.ema_val * ema_weight + (1 - self.ema_val) * weight)

        ram_usage = float(get_ram_info()["percentage"])

        return {m.name: m.result() for m in self.metrics} | {"RAM %": ram_usage}





    def test_step(self, images):
        if self.latent_diffusion:
            images = self.image_encoder(images)
        batch_size, image_height, image_width, n_channels = ops.shape(images)

        noises = keras.random.normal(
            shape=(batch_size, image_height, image_width, n_channels)
        )

        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        ram_usage = float(get_ram_info()["percentage"])
        return {m.name: m.result() for m in self.metrics} | {"RAM %": ram_usage}






    def active_sampling(
        self,
        image,
        initial_operator,
        update_operator_fn,
        num_samples_to_take,
        sampling_window,
        train_h_config,
        guidance_method="dps",
        initial_samples=None,
        initial_step=0,
        image_shape=None,
        diffusion_steps=None,
        guidance_kwargs=None,
        verbose=False,
        plot_callback=None,
        plotting_interval=50,
    ):
        import cv2

        print(f"num sample to take ddim: {num_samples_to_take}")
        if image_shape is None:
            image_shape = ops.shape(image)
        # noise -> images -> denormalized images
        if diffusion_steps is None:
            diffusion_steps = self.diffusion_steps

        assert (
            len(image_shape) == 4
        ), "image_shape must be a tuple of (batch, height, width, channels)"

        num_images, image_height, image_width, n_channels = image_shape

        if self.latent_shape:
            image_height = image_height // 8
            image_width = image_width // 8
            n_channels = 4

        initial_noise = keras.random.normal(
            shape=(num_images, image_height, image_width, n_channels)
        )

        num_images = ops.shape(initial_noise)[0]
        step_size = 1.0 / diffusion_steps

        if guidance_kwargs is None:
            guidance_kwargs = {"omega": 10}

        # start the reverse conditional diffusion process
        progbar = keras.utils.Progbar(diffusion_steps, verbose=verbose)

        operator = initial_operator
        start_sampling, stop_sampling = sampling_window
        assert start_sampling >= initial_step, (
            "sampling_window = (start, stop), where start should be >= initial_step, "
            f"got sampling_window = {sampling_window}"
        )
        assert stop_sampling <= diffusion_steps, (
            "sampling_window = (start, stop), where stop should be <= diffusion_steps, "
            f"got sampling_window = {sampling_window}"
        )
        sampling_interval = (stop_sampling - start_sampling) // num_samples_to_take
        print(f"ddim ampling interval: {sampling_interval}")
        sampling_interval = max(sampling_interval, 1)
        measurements = operator.forward(image)
        run_guidance = self.get_guidance_fn(
            guidance_method,
            operator,
        )

        if initial_samples is not None and initial_step > 0:
            starting_diffusion_times = ops.ones((num_images, 1, 1, 1)) - (
                (initial_step - 1) * step_size
            )
            noise_rates, signal_rates = self.diffusion_schedule(
                starting_diffusion_times
            )
            next_noisy_images = (
                signal_rates * initial_samples + noise_rates * initial_noise
            )
        else:
            next_noisy_images = initial_noise


        ori_img = np.array(image[0])
        ori_img = ((ori_img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        ori_img = ori_img.astype(np.uint8)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"masks/ori_img.png", ori_img)
        cnt = 0

        for step in range(initial_step, diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )


            next_noisy_images, measurement_error, pred_images = run_guidance(
                noisy_images,
                measurements,
                noise_rates,
                signal_rates,
                next_signal_rates,
                next_noise_rates,
                **guidance_kwargs,
            )

            if self.latent_diffusion:
                print("latent!")
                pred_images = self.decoder(pred_images)
            

            if (
                step >= start_sampling
                and step < stop_sampling
                and step % sampling_interval == 0
            ):


                images = np.array(pred_images)
                images = ((images + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                #images = np.clip(images, 0, 1) * 255
                images = images.astype(np.uint8)

                #save_path = os.path.join("temp_tt", f"step_{step}.png")
                #save_image_grid(images, save_path, grid_size=(4, 4))


                operator = update_operator_fn(
                    pred_images, step = step, sampling_window=sampling_window, sampling_interval=sampling_interval
                )
                
                measurements = operator.forward(image)
             
                cv2.imwrite(f"masks/mask_{step}.png", np.array(operator.mask[0])*255)
                #pos_path = os.path.join("vis_pos", f"step_{step}.png")

                #masked_path = os.path.join("par_obs", f"step_{step}.png")
                #image_store = ((image[0]+1) * 127.5).astype(np.uint8)
                #image_masked_store = np.array(image_store*operator.mask[0])
                cnt+=1
                
                 
                
                if cnt%20 == 0:
                    env = os.environ.copy()
                    env["KERAS_BACKEND"] = "tensorflow"
                
                    subprocess.run(
                        [
                            "python", "train_ptdm.py",
                            "--train_config", "configs/training/ptdm_imgnet.yaml",    
                            "--data_root", "data/",
                            "--mask_path",f"masks/mask_{step}.png",
                            "--image_path","masks/ori_img.png",
                        ],
                        env=env
                    )
                    self.h_trans_model = load_model(
                        Path(str(train_h_config.h_model_run_dir)) / "checkpoints",
                        stable_diffusion_kwargs={
                            "img_height": image_shape[1],
                            "img_width": image_shape[2],
                        },
                        image_shape=image_shape[-3:],
                    )
                    run_guidance = self.get_guidance_fn(guidance_method, operator)
                
                



            elif plotting_interval is not None and step % plotting_interval == 0:
                if verbose and plot_callback is not None:
                    plot_callback.add_to_buffer(
                        step, operator.mask, pred_images, noisy_images
                    )

            # this new noisy image will be used in the next step
            progbar.update(step + 1, [("error", measurement_error)])

        return pred_images, measurements, operator#, pred_images_100, measurements_100,pred_images_200, measurements_200


    @property
    def decoder(self):
        """decoder returns the diffusion image decoder model with pretrained
        weights. Can be overriden for tasks where the decoder needs to be
        modified.
        """
        return self.autoencoder.decode

    @property
    def image_encoder(self):
        """image_encoder returns the autoencoder Encoder with pretrained weights."""
        return self.autoencoder.encode

    def save_model_json(self, directory):
        """Save model as JSON file."""
        json_model = self.to_json()
        json_model_path = str(Path(directory) / "model.json")
        with open(json_model_path, "w", encoding="utf-8") as json_file:
            json_file.write(json_model)
        log.info(f"Succesfully saved model architecture to {json_model_path}")

    def load_weights(self, filepath, *args, **kwargs):

        with h5py.File(filepath, "r") as f:
            if "layers/vae_model" not in f:
                super().load_weights(filepath, *args, **kwargs)
            else:
                assert "layers/functional" in f, (
                    "The weights file must contain the 'layers/functional' group "
                    "to load the model weights"
                )
                temp_file = Path("temp.weights.h5")
                with h5py.File(temp_file, "w") as f_temp:
                    # copy layers/functional from f to f_temp
                    f.copy("layers/functional", f_temp, "layers/functional")
                    f.copy("vars", f_temp, "vars")
                    f.copy("optimizer", f_temp, "optimizer")
                    f.copy("ema_network", f_temp, "ema_network")
                    # copy layers/vae_model from f to f_temp
                    f.copy("layers/vae_model", f_temp, "autoencoder")
                super().load_weights(str(temp_file), *args, **kwargs)
                # delete temp file
                temp_file.unlink()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,

                "mean": self.mean,
                "variance": self.variance,
                "image_range": self.image_range,
                "latent_diffusion": self.latent_diffusion,
                "latent_shape": self.latent_shape,
                "autoencoder_checkpoint_directory": self.autoencoder_checkpoint_directory,
            }
        )
        return config
