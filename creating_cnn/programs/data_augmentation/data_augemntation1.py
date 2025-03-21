import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm
import random

# Example Usage
input_folder = "creating_training_set/augmentation_sample/augmentation_images"
target_folder = "creating_training_set/augmentation_sample/augmentatin_targets"
output_folder_images = "creating_training_set/augmentation_sample/augmentation_image_results"
output_folder_targets = "creating_training_set/augmentation_sample/augmentation_targets_results"


def augment_images(input_folder, target_folder, output_folder_images, output_folder_targets, num_augmentations=5):
    """
    Applies identical data augmentations to images and their corresponding targets.

    """
    
    # Create output directories if they don't exist
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_targets, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for img_name in tqdm(image_files):
        img_path = os.path.join(input_folder, img_name)
        target_path = os.path.join(target_folder, img_name)

        # Ensure the target exists
        if not os.path.exists(target_path):
            print(f"Warning: No target found for {img_name}, skipping...")
            continue
        
        # Read image and target
        img = read_image(img_path).float() / 255.0  # Normalize to [0,1]
        target = read_image(target_path).float() / 255.0  # Normalize target as well

        # Convert RGBA to RGB if needed
        if img.shape[0] == 4:
            img = img[:3, :, :]
        if target.shape[0] == 4:
            target = target[:3, :, :]

        for i in range(num_augmentations):
            # Generate random transformation parameters
            angle = random.uniform(-15, 15)
            hflip = random.random() > 0.5
            scale = random.uniform(0.8, 1.0)
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.8, 1.0), ratio=(0.75, 1.33))

            # Apply transformations consistently
            img_aug = TF.rotate(img, angle)
            target_aug = TF.rotate(target, angle)

            if hflip:
                img_aug = TF.hflip(img_aug)
                target_aug = TF.hflip(target_aug)

            img_aug = TF.resized_crop(img_aug, i, j, h, w, size=(256, 256))
            target_aug = TF.resized_crop(target_aug, i, j, h, w, size=(256, 256))

            img_aug = TF.adjust_brightness(img_aug, brightness_factor=random.uniform(0.8, 1.2))
            img_aug = TF.adjust_contrast(img_aug, contrast_factor=random.uniform(0.8, 1.2))

            # Save augmented images with the same filename
            save_image(img_aug, os.path.join(output_folder_images, f"aug_{i}_{img_name}"))
            save_image(target_aug, os.path.join(output_folder_targets, f"aug_{i}_{img_name}"))
    
    print(f"Augmentation complete! Augmented images saved in {output_folder_images} and {output_folder_targets}")



augment_images(input_folder, target_folder, output_folder_images, output_folder_targets)
