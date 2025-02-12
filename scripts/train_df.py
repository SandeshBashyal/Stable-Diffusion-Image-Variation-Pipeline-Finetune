from diffusers import StableDiffusionImageVariationPipeline
import torch
from torch.optim import AdamW
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from datasets import load_dataset
import torchvision
from argparse import Namespace

import math
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionImageVariationPipeline
from torch.optim.lr_scheduler import CosineAnnealingLR
from lpips import LPIPS

from load_data import CustomImageDataset
from utils import collate_fn, calculate_psnr


def training_function_single_epoch(args, vae, unet, clip_encoder, feature_extractor):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",  # Enable mixed precision
    )

    set_seed(args.seed)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        unet.parameters(),  # Only optimize unet
        lr=args.learning_rate,
    )

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_train_steps, eta_min=1e-7)
    
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    train_dataloader = DataLoader(
        args.train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # Move vae and clip_encoder to the accelerator device
    vae.to(accelerator.device)
    clip_encoder.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["instance_images"].to(accelerator.device)).latent_dist.sample()
                    latents = latents * 0.18215  # Scale the latents (VAE scaling factor)

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the CLIP embeddings for conditioning
                clip_embeddings = batch["clip_embeddings"].to(accelerator.device)
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=clip_embeddings).sample

                # Calculate the loss
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item() * args.gradient_accumulation_steps}  # Rescale loss for logging
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using the trained modules and save it
    if accelerator.is_main_process:
        print(f"Loading pipeline and saving to {args.output_dir}...")
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
        )
        pipeline = StableDiffusionImageVariationPipeline(
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            scheduler=scheduler,
            image_encoder=clip_encoder,
            safety_checker= None,
            feature_extractor=feature_extractor
        )
        pipeline.save_pretrained(args.output_dir)
        
from torch.utils.tensorboard import SummaryWriter  # Optional: For logging

def training_function(args, vae, unet, clip_encoder, feature_extractor):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",  # Enable mixed precision
    )

    set_seed(args.seed)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        unet.parameters(),  # Only optimize unet
        lr=args.learning_rate,
    )

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_train_steps, eta_min=1e-7)
    
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    train_dataloader = DataLoader(
        args.train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # Move vae and clip_encoder to the accelerator device
    vae.to(accelerator.device)
    clip_encoder.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Optional: TensorBoard logging
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break  # Stop after max_train_steps

            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["instance_images"].to(accelerator.device)).latent_dist.sample()
                    latents = latents * 0.18215  # Scale the latents (VAE scaling factor)

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the CLIP embeddings for conditioning
                clip_embeddings = batch["clip_embeddings"].to(accelerator.device)
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=clip_embeddings).sample

                # Calculate the loss
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Decode the predicted latents back to images
            with torch.no_grad():
                pred_latents = (noisy_latents - noise_pred) / noise_scheduler.init_noise_sigma  # Reverse the noise addition
                pred_images = vae.decode(pred_latents / 0.18215).sample  # Decode latents to images
                pred_images = torch.clamp(pred_images, -1, 1)  # Clamp to valid image range
                pred_images = (pred_images + 1) / 2  # Normalize to [0, 1] for PSNR calculation

                # Ground truth images (normalized to [0, 1])
                gt_images = batch["instance_images"].to(accelerator.device)
                gt_images = (gt_images + 1) / 2  # Normalize to [0, 1]

                # Calculate PSNR
                psnr = calculate_psnr(pred_images, gt_images)

            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item() * args.gradient_accumulation_steps,  # Rescale loss for logging
                "psnr": psnr,  # Log PSNR
            }
            progress_bar.set_postfix(**logs)

            # Log to TensorBoard (optional)
            writer.add_scalar("Loss/train", logs["loss"], global_step)
            writer.add_scalar("PSNR/train", logs["psnr"], global_step)

            # Save checkpoint every 4000 steps
            if global_step % 4000 == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Save model, optimizer, and scheduler states
                accelerator.save_state(checkpoint_dir)
                print(f"Checkpoint saved at step {global_step} to {checkpoint_dir}")

        accelerator.wait_for_everyone()

    # Final checkpoint at the end of training
    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        accelerator.save_state(checkpoint_dir)
        print(f"Final checkpoint saved at step {global_step} to {checkpoint_dir}")

        # Save the final pipeline
        print(f"Loading pipeline and saving to {args.output_dir}...")
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
        )
        pipeline = StableDiffusionImageVariationPipeline(
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            scheduler=scheduler,
            image_encoder=clip_encoder,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(args.output_dir)

    # Close TensorBoard writer
    writer.close()