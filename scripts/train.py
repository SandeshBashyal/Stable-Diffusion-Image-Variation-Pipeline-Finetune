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
from train_df import training_function

learning_rate = 2e-06
max_train_steps = 400
batch_size = 1

# Load the pipeline and components
pipeline = StableDiffusionImageVariationPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers",
                                                                 requires_safety_checker=False,
                                                                 device='cuda')
pipeline.enable_model_cpu_offload()
unet = pipeline.unet
vae = pipeline.vae
clip_encoder = pipeline.image_encoder.to('cpu')
# feature_extractor = pipeline.feature_extractor.to('cpu')
# Freeze the VAE and CLIP encoder
for param in vae.parameters():
    param.requires_grad = False
for param in clip_encoder.parameters():
    param.requires_grad = False

# Optimizer (only train the UNet)
path_file = '/home/rmuproject/rmuproject/data/**/*.png'
path_list = glob.glob(path_file, recursive=True)

feature_extractor = pipeline.feature_extractor
# Create the dataset
dataset_name = "/home/rmuproject/rmuproject/data"  # @param
dataset = load_dataset(dataset_name, split="train")

image_variation_dataset = CustomImageDataset(dataset, clip_encoder, feature_extractor)

# Assuming `dataset` is your ImageVariationDataset
dataloader = DataLoader(image_variation_dataset, batch_size= batch_size, shuffle=True, collate_fn=collate_fn)

args = Namespace(
    # pretrained_model_name_or_path=model_id,
    resolution=512,  # Reduce this if you want to save some memory
    train_dataset=image_variation_dataset,
    checkpointing_steps=4000,  # Save a checkpoint every 4000 steps
    resume_from_checkpoint=None,  # Set to a checkpoint path to resume training
    max_train_steps = num_train_epochs * steps_per_epoch, 
    # instance_prompt=instance_prompt,
    learning_rate=learning_rate,
    train_batch_size=1,
    gradient_accumulation_steps=1,  # Increase this if you want to lower memory usage
    max_grad_norm=1.0,
    gradient_checkpointing=True,  # Set this to True to lower the memory usage
    use_8bit_adam=True,  # Use 8bit optimizer from bitsandbytes
    seed=3434554,
    sample_batch_size=2,
    output_dir="/home/rmuproject/rmuproject/users/sandesh/models/50_epochs/",  # Where to save the pipeline
)

training_function(args, vae, unet, clip_encoder, feature_extractor)