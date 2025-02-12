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

from utils import collate_fn
# from train_df import training_function

from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataset, clip_encoder, feature_extractor, size=512):
        """
        Args:
            dataset: A dataset object from the `datasets` library.
            clip_encoder: The CLIP image encoder model (e.g., CLIPVisionModelWithProjection).
            feature_extractor: The feature extractor from the StableDiffusionImageVariationPipeline.
            size: The size to which images should be resized for the UNet (default: 512x512).
        """
        self.dataset = dataset
        self.clip_encoder = clip_encoder
        self.feature_extractor = feature_extractor
        self.size = size

        # Transformations for the input images (resize, normalize, etc.)
        self.unet_transforms = transforms.Compose(
            [
                transforms.Resize((size, size)),  # Resize to the required size for UNet
                transforms.CenterCrop(size),      # Center crop to ensure square images
                transforms.ToTensor(),            # Convert to tensor
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )

        self.clip_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize to 224x224 for CLIP
                transforms.ToTensor(),          # Convert to tensor
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        try:
            # Load the image
            image = self.dataset[index]["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")  # Ensure it's a PIL image

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Transform the image for the UNet
            example["instance_images"] = self.unet_transforms(image)

            # Transform the image for CLIP
            clip_image = self.clip_transforms(image)

            # Generate the CLIP embedding
            with torch.no_grad():
                # Preprocess the image for CLIP using the feature_extractor
                clip_input = clip_image.unsqueeze(0).to(self.clip_encoder.device)  # Shape: [1, 3, 224, 224]
                clip_embedding = self.clip_encoder(clip_input).image_embeds  # Shape: [1, embedding_dim]

            # Add a sequence length dimension to the CLIP embedding
            clip_embedding = clip_embedding.unsqueeze(1).to('cpu')  # Shape: [1, 1, embedding_dim]
            example["clip_embeddings"] = clip_embedding.squeeze(0)  # Remove batch dimension for collation

        except Exception as e:
            # Skip corrupted or invalid images
            print(f"Error processing image at index {index}: {e}")
            return self.__getitem__((index + 1) % len(self))  # Skip to the next image

        return example
    
class CustomImageDataset_FeatureExtract(Dataset):
    def __init__(self, dataset, clip_encoder, feature_extractor, size=512):
        """
        Args:
            dataset: A dataset object from the `datasets` library.
            clip_encoder: The CLIP image encoder model (e.g., CLIPVisionModelWithProjection).
            feature_extractor: The feature extractor from the StableDiffusionImageVariationPipeline.
            size: The size to which images should be resized (default: 512x512).
        """
        self.dataset = dataset
        self.clip_encoder = clip_encoder
        self.feature_extractor = feature_extractor
        self.size = size

        # Transformations for the input images (resize, normalize, etc.)
        self.transforms = transforms.Compose(
            [
                transforms.Resize((size, size)),  # Resize to the required size
                transforms.CenterCrop(size),      # Center crop to ensure square images
                transforms.ToTensor(),            # Convert to tensor
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),            # Data augmentation
                # transforms.RandomHorizontalFlip(p=0.5),            # Data augmentation
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        try:
            # Load the image
            image = self.dataset[index]["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")  # Ensure it's a PIL image

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Transform the image for the UNet
            example["instance_images"] = self.transforms(image)

            # Generate the CLIP embedding
            with torch.no_grad():
                # Preprocess the image for CLIP using the feature_extractor
                clip_input = self.feature_extractor(image, return_tensors="pt").pixel_values.squeeze(0)  # Shape: [3, 224, 224]
                clip_input = clip_input.to(self.clip_encoder.device)  # Move to the correct device
                clip_embedding = self.clip_encoder(clip_input.unsqueeze(0)).image_embeds  # Shape: [1, embedding_dim]

            # Add a sequence length dimension to the CLIP embedding
            clip_embedding = clip_embedding.unsqueeze(1).to('cpu')  # Shape: [1, 1, embedding_dim]
            example["clip_embeddings"] = clip_embedding.squeeze(0)  # Remove batch dimension for collation

        except Exception as e:
            # Skip corrupted or invalid images
            print(f"Error processing image at index {index}: {e}")
            return self.__getitem__((index + 1) % len(self))  # Skip to the next image

        return example  