import torch

def collate_fn(examples):
    """
    Collate function for the ImageVariationDataset.
    Args:
        examples: A list of dictionaries, where each dictionary contains:
            - "instance_images": A tensor of shape [C, H, W].
            - "clip_embeddings": A tensor of shape [embedding_dim].
    Returns:
        A dictionary containing:
            - "instance_images": A stacked tensor of shape [batch_size, C, H, W].
            - "clip_embeddings": A stacked tensor of shape [batch_size, embedding_dim].
    """
    # Extract instance_images and clip_embeddings from the examples
    instance_images = [example["instance_images"] for example in examples]
    clip_embeddings = [example["clip_embeddings"] for example in examples]

    # Stack the tensors along the batch dimension
    instance_images = torch.stack(instance_images)
    clip_embeddings = torch.stack(clip_embeddings)

    # Ensure the instance_images tensor is in contiguous memory format and cast to float
    instance_images = instance_images.to(memory_format=torch.contiguous_format).float()

    # Return the batch as a dictionary
    batch = {
        "instance_images": instance_images,  # Shape: [batch_size, C, H, W]
        "clip_embeddings": clip_embeddings,  # Shape: [batch_size, embedding_dim]
    }
    return batch

import torch.nn.functional as F
def calculate_psnr(pred, target):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.
    Assumes images are in the range [0, 1].
    """
    mse = F.mse_loss(pred, target)  # Mean Squared Error
    if mse == 0:  # PSNR is infinite if there's no noise
        return float('inf')
    max_pixel = 1.0  # Assuming images are normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()