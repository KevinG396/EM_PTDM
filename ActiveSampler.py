# pylint: disable=no-member

import os
import uuid
from abc import ABC, abstractmethod
from itertools import islice

import numpy as np
import utils.lib.utils as lib_utils
from keras import ops
from PIL import Image
from utils.lib import log
from utils.lib.config import load_config_from_yaml
import jax.numpy as jnp

from PIL import Image
from datasets import get_datasets
#from utils import mri
from load_model import load_model
from measurements import prepare_measurement
from ddim import DiffusionModel
from h_trans import HTransformModel

import tensorflow as tf
from utils.keras_utils import (
    load_img_as_tensor,
    normalize,
    postprocess_image,
    translate,
)

from pix_nn import (
    #TargetClassifier,
    TargetConvResizer,

)

def get_active_diffusion_sampler_class(data_domain):
    if data_domain == "image":
        return ImageActiveSampler

    else:
        raise ValueError(f"data domain `{data_domain}` was not recognised.")




class ActiveSampler(ABC):

    def __init__(
        self,
        image_shape,

        diffusion_model_run_dir,   # here!
        target_img_path,
        mask_img_path,
        selection_strategy,
        config,
        hard_consistency=False,
        initial_measurement=False,
        sigma=1,
        pixel_region_radius=None,
        data_root=None,
    ):

        self.block_size = 4
        self.selection_strategy = selection_strategy
        self.image_shape = image_shape
        self.pixel_region_radius = pixel_region_radius
        self.data_root = data_root
        self.mask_img_path = mask_img_path

        self.diffusion_model_run_dir = diffusion_model_run_dir

        
        patch_size = self.block_size
        print(f"patch sizzeee {patch_size}")
        self.target_model = TargetConvResizer(patch_size, self.image_shape[-1])

        if self.pixel_region_radius is not None:
            assert (
                self.selection_strategy == "pixel_variance"
            ), "pixel_region_radius is only supported for pixel_variance selection strategy."

        

        if str(diffusion_model_run_dir) == "stable_diffusion":
            self.diffusion_config = load_config_from_yaml(
                "configs/stable_diffusion/config.yaml"
            )
            log.info(
                "Loading StableDiffusion model with parameters from "
                f"{log.yellow('configs/stable_diffusion/config.yaml')}."
            )
        else: ## here!!!
            self.diffusion_config = load_config_from_yaml(
                diffusion_model_run_dir / "config.yaml"
            )



        self.main_diffusion_model = load_model(
            diffusion_model_run_dir / "checkpoints",
            stable_diffusion_kwargs={
                "img_height": image_shape[0],
                "img_width": image_shape[1],
            },
            image_shape=image_shape,
        )

        print("main model successfully loaded\n")

                
        self.h_diffusion_model = DiffusionModel(
            image_shape = config.data.image_shape,
            widths =config.model.widths,
            block_depth = config.model.block_depth,
            ema_val=config.optimization.ema,
            min_signal_rate=config.sampling.min_signal_rate,
            max_signal_rate=config.sampling.max_signal_rate,
            diffusion_steps=config.model.diffusion_steps,
            image_range=config.data.normalization,

            latent_diffusion=config.model.latent,
            latent_shape=config.model.latent_shape if config.model.latent else None,
            autoencoder_checkpoint_directory=config.model.get(
                "autoencoder_checkpoint_directory"
            ),
        )
        
        
        


        print("h model successfully loaded\n")

        self.diffusion_model = HTransformModel(
            image_shape,
            self.main_diffusion_model,
            self.h_diffusion_model
        )

        self.target_img = self.load_target_img(target_img_path)
        print(f"target_img: {self.target_img.shape}")

        datestring = lib_utils.get_date_string()

        self.save_dir = (
            diffusion_model_run_dir / "inference" / (datestring + "_active_sampling")
        )
        self.save_dir.mkdir(parents=True)
        self.initial_measurement = initial_measurement
        self.operator, self.measurement = self.initialise_operator()
        self.hard_consistency = hard_consistency

        if selection_strategy == "pixel_random":
            self.measurement_selection_fn = None
        elif selection_strategy == "pixel_data_variance":
            self.measurement_selection_fn = None
        elif selection_strategy == "pixel_variance":
            self.measurement_selection_fn = self.select_pixel_variance
        elif selection_strategy == "column_variance":
            self.measurement_selection_fn = self.select_column_variance
        elif selection_strategy == "column_random":
            self.measurement_selection_fn = None
        elif selection_strategy == "column_data_variance":
            self.measurement_selection_fn = None

        # pixel selection
        elif selection_strategy == "pixel_entropy":
            self.measurement_selection_fn = lambda x,step: self.select_pixel_entropy(
                x, step, sigma=sigma
            )
        elif selection_strategy == "block":
            self.measurement_selection_fn = lambda x,step,sampling_window,sampling_interval: self.select_block_entropy(
                x, step,sampling_window,sampling_interval, sigma=sigma
            )
        elif selection_strategy == "column_entropy":
            self.measurement_selection_fn = lambda x: self.select_column_entropy(
                x, sigma=sigma
            )
        elif selection_strategy == "column_equispaced":
            self.measurement_selection_fn = None
        elif selection_strategy == "fastmri_baseline":
            self.measurement_selection_fn = None
        else:
            raise ValueError(
                f"selection_strategy={selection_strategy} is not a valid option."
            )

    @abstractmethod
    def initialise_operator(self):

        return


    def select_block_entropy(self, measurement_particles, step,sampling_window, sampling_interval, sigma=1):
        block_size = self.block_size

        error_matrices = ops.convert_to_tensor(
            [
                ops.convert_to_tensor(
                    [(particle_i - particle_j) for particle_i in measurement_particles]
                )
                for particle_j in measurement_particles
            ]
        )
        

        measurement_particles_np = np.array(measurement_particles)
        target_img = self.target_img
        target_img_tf = tf.convert_to_tensor(target_img)
        measurement_particles_tf = tf.convert_to_tensor(measurement_particles_np)
        target_model = self.target_model

        
        
        N_size, inp_dim_1, inp_dim_2, num_channels = measurement_particles.shape

        def split_patches(image, patch_size):
            
            B, H, W, C = image.shape
            assert H % patch_size == 0 and W % patch_size == 0

            patches = tf.reshape(image, (B, H // patch_size, patch_size, W // patch_size, patch_size, C))
            patches = tf.transpose(patches, perm=[0, 1, 3, 2, 4, 5])
            return patches

        patches = split_patches(measurement_particles_tf, block_size)
        
        
        B, H, W, C = measurement_particles.shape  
        block_reward = tf.zeros((B, H // block_size, W // block_size))

        patches = tf.reshape(
            measurement_particles, 
            (B, H // block_size, block_size, W // block_size, block_size, C)
        )
        patches = tf.transpose(patches, perm=[0, 1, 3, 2, 4, 5])  
         
        
        B_new = B * (H // block_size) * (W // block_size)
        reshaped_patches = tf.reshape(patches, (B_new, block_size, block_size, C))  

        output = target_model(reshaped_patches) 

        softmax_output = tf.nn.softmax(output, axis=-1)  
        rewards_flat = softmax_output[:, 0]  

        block_reward = tf.reshape(rewards_flat, (B, H // block_size, W // block_size))
        

        start_step = sampling_window[0]
        end_step = sampling_window[1]
        budget = int(((end_step-start_step)/sampling_interval))
        b_curr = int(((step-start_step)/sampling_interval) + 1)
        

        
        alpha_1 = (budget - b_curr)/(budget + b_curr)
        if (alpha_1 < 0):
            alpha_1 = 0
        
        alpha_2 = 1 - alpha_1
        squared_l2_per_pixel_i_j = ops.sum(error_matrices**2, axis=[-1])

        exploration = ops.exp(
            (squared_l2_per_pixel_i_j) / (2 * sigma**2)
        )


        likelihood = ops.exp(
            -(squared_l2_per_pixel_i_j) / (2 * sigma**2)
        )


        gaussian_error_per_pixel_i_j = exploration




        confidence_per_pixel_i = ops.sum(likelihood, axis=1)

        entropy_per_pixel_i = ops.sum(gaussian_error_per_pixel_i_j, axis=1)

        def block_average(tensor, block_size):

            batch_size, h, w = tensor.shape
            h_new, w_new = h // block_size * block_size, w // block_size * block_size
            tensor = tensor[:, :h_new, :w_new]
            tensor = tensor.reshape(
                batch_size, 
                h_new // block_size, block_size,
                w_new // block_size, block_size
            )
            return tensor.mean(axis=(2, 4))


        
        likelihood_per_block = block_average(confidence_per_pixel_i, block_size)
        
        exploration_per_block = block_average(entropy_per_pixel_i, block_size)
       
        
        exploitation_per_block = block_reward * likelihood_per_block
        
        entropy_per_block = alpha_1 * exploration_per_block + alpha_2 * exploitation_per_block

        entropy_per_block_log = ops.sum(ops.log(entropy_per_block), axis=0)#ops.log(entropy_per_block)

        entropy_per_block_log = (
            entropy_per_block_log * ops.logical_not(self.operator.block_mask)
        )

        block_index = np.unravel_index(
            ops.argmax(entropy_per_block_log), shape=self.operator.block_mask.shape
        )
        return block_index


    def preprocess(self, x):
        """
        """
        return x

    def postprocess(self, x):
        """
        """
        return x

    def sample_and_reconstruct(
        self,
        num_samples_to_take,
        sampling_window,
        posterior_shape,
        num_diffusion_steps, 
        train_h_config,
        guidance_kwargs=None,
        guidance_method="dps",
        verbose=True,
        plot_callback=None,
        plotting_interval=None,
    ):


        posterior_samples, measurements, _ = self.diffusion_model.active_sampling(

            self.target_img,
            self.operator,
            self.update_operator,
            num_samples_to_take=num_samples_to_take,
            sampling_window=sampling_window,
            image_shape=posterior_shape,
            diffusion_steps=num_diffusion_steps,
            guidance_method=guidance_method,
            guidance_kwargs=guidance_kwargs,
            train_h_config = train_h_config,
            verbose=verbose,
            plot_callback=plot_callback,
            plotting_interval=plotting_interval,
        )
        if verbose:
            mae = ops.mean(
                ops.abs(self.target_img - ops.mean(posterior_samples, axis=0))
            )
            log.info(f"MAE: {mae:.4f}")

        return posterior_samples, measurements

    def save_result(self, posterior_samples, measurements):

        posterior_mean = ops.mean(posterior_samples, axis=0)
        posterior_mean = self.postprocess(posterior_mean)

        posterior_samples = self.postprocess(posterior_samples)
        measurements = self.postprocess(measurements)
        mask_ori = self.operator.mask
        mask = mask_ori * 255

        target = self.postprocess(self.target_img)

        search_result = mask_ori * target

        images = {
            "posterior_mean": posterior_mean,
            "target": target,
            "mask": mask,
            "measurement": measurements,
            "search": search_result,
            **{
                f"posterior_sample_{i}": posterior_sample
                for i, posterior_sample in enumerate(posterior_samples)
            },
        }

        for key, image in images.items():
            path = (self.save_dir / key).with_suffix(".png")
            image = ops.convert_to_numpy(image)
            image = np.squeeze(image).astype("uint8")
            Image.fromarray(image).save(path)
            log.info(f"Saved {key} to {path}")
        return images


class ImageActiveSampler(ActiveSampler):


    def load_target_img(self, target_img_path):

        if target_img_path.startswith("validation_dataset_"):
            raise UserWarning(
                "Loading validation set images is currently only implemented for fastMRI. Please specify an absolute path for other datasets."
            )
        if target_img_path == "benchmark":
            return ops.zeros(self.image_shape)[None, ...]
        else:
            target_img = load_img_as_tensor(
                str(target_img_path),
                image_shape=self.image_shape[:2],
                grayscale=bool(self.image_shape[-1] == 1),
            )
            target_img = ops.expand_dims(target_img, axis=0)
            log.info(f"Loaded target image from {log.yellow(target_img_path)}")
            return self.preprocess(target_img)

    def initialise_operator(self):

        if self.initial_measurement is True:
            raise NotImplementedError(
                "Smart initial measurements have not yet ben implemented for image domain. Please set self.initial_measurement=False"
            )
        initial_mask = ops.zeros(self.target_img.shape)
        initial_block_mask = ops.zeros((self.target_img.shape[1]//self.block_size,self.target_img.shape[2]//self.block_size))

        operator, measurement = prepare_measurement(
            "inpainting",
            ops.convert_to_tensor(
                self.target_img
            ),  
            mask=initial_mask,
            block_mask = initial_block_mask,
        )
        return operator, measurement




    def update_operator(self, pred_images, step, sampling_window, sampling_interval):

        particles_in_measurement_space = pred_images
        selected_pixel = self.measurement_selection_fn(
            particles_in_measurement_space, step = step, sampling_window = sampling_window, sampling_interval=sampling_interval
        )
        self.operator.block_mask = self.operator.block_mask.at[selected_pixel].set(1)
        batch, channel = self.target_img.shape[0], self.target_img.shape[3]
            

        row_block, col_block = selected_pixel
        row_start = int(row_block) * self.block_size
        col_start = int(col_block) * self.block_size
            
        original_pixels = [
            (b, row, col, c)
            for b in range(batch)
            for row in range(row_start, row_start + self.block_size)
            for col in range(col_start, col_start + self.block_size)
            for c in range(channel)
        ]
            
            
        if pred_images.shape[-1]==1:
                
            for original_pixel in original_pixels:
                self.operator.mask = self.operator.mask.at[original_pixel].set(1)
                #np.set_printoptions(threshold=np.inf)
                
        elif pred_images.shape[-1]!=1:
            for original_pixel in original_pixels:
                    #for c in range(self.operator.mask.shape[-1]):  
                self.operator.mask = self.operator.mask.at[original_pixel].set(1)
                #print(f"mask shape{self.operator.mask.shape}")
            


             
        
        def load_mask(mask_path):

            mask = load_img_as_tensor(
                str(mask_path),
                image_shape=self.image_shape[:2],
                grayscale=bool(self.image_shape[-1] == 1),
            )
                
            gray_mask = 0.299 * mask[:, :, 0] + 0.587 * mask[:, :, 1] + 0.114 * mask[:, :, 2]
            binary_mask = jnp.where(gray_mask >= 0.7, 1, 0)
            mask = np.array(binary_mask)
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
                
            return mask

        gt_mask = load_mask(self.mask_img_path)
            # gt_mask.reshape(self.target_img.shape[1], self.target_img.shape[2]) # torch
        gt_mask = tf.reshape(gt_mask, (self.target_img.shape[1], self.target_img.shape[2]))
           
            
        def create_block_training_set(image, mask_img, block_mask, patch_size):
            image = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
            mask_img = tf.convert_to_tensor(mask_img, dtype=tf.float32)
            block_mask = tf.convert_to_tensor(block_mask, dtype=tf.float32)
            if image.shape[0] == 1:
                image = tf.squeeze(image, axis=0)
            patches = tf.image.extract_patches(
                images=tf.expand_dims(image, axis=0),
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID"
            )

            mask_patches = tf.image.extract_patches(
                images=tf.expand_dims(mask_img, axis=0)[..., tf.newaxis],
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID"
            )

            mask_flat = tf.reshape(block_mask, [-1])
            valid_indices = tf.where(mask_flat == 1)[:, 0]
            patches = tf.gather(tf.reshape(patches, [-1, patch_size, patch_size, tf.shape(image)[-1]]), valid_indices)
            mask_patches = tf.gather(tf.reshape(mask_patches, [-1, patch_size, patch_size]), valid_indices)

            avg_values = tf.reduce_mean(mask_patches, axis=[1, 2])
            labels = tf.stack([avg_values, 1 - avg_values], axis=1)

            return patches, labels
               
            
        train_data, labels = create_block_training_set(self.target_img, gt_mask, self.operator.block_mask, self.block_size)
        
        def train_simple(model, patches, labels, epochs=10, lr=0.001):
   
            optimizer = tf.optimizers.Adam(learning_rate=lr)
            loss_fn = tf.nn.softmax_cross_entropy_with_logits

            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    logits = model(patches)  
                    loss = tf.reduce_mean(loss_fn(labels=labels, logits=logits))  

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_simple(
            model=self.target_model, 
            patches=train_data, 
            labels=labels, 
            epochs=3, 
            lr=0.001
        )
            

            
            
        return self.operator


    def preprocess(self, x):

        return translate(
            x,
            (0, 255),
            self.diffusion_model.image_range,
        )

    def postprocess(self, x):

        return postprocess_image(
            x,
            self.diffusion_config.data.normalization,
        )

