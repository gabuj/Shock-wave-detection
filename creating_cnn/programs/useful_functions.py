import torch
import torch.nn.functional as F

def collate_fn(batch):
    """
    Custom collate function to pad images and labels in a batch to the same size.
    """
    images, masks = zip(*batch)  # Unzip images and corresponding masks

    # Find max height and width in the batch
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    # Pad all images and masks to the same size
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in images]
    padded_masks = [F.pad(mask, (0, max_width - mask.shape[2], 0, max_height - mask.shape[1])) for mask in masks]

    # Stack to create batch
    return torch.stack(padded_images), torch.stack(padded_masks)