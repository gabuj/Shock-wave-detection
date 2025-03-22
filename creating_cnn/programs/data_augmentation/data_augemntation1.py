import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm
import random


num_augmentations=10


# Example Usage
input_folder = "creating_training_set/augmentation_sample/augmentation_images"
target_folder = "creating_training_set/augmentation_sample/augmentatin_targets"
output_folder_images = "creating_training_set/augmentation_sample/augmentation_image_results"
output_folder_targets = "creating_training_set/augmentation_sample/augmentation_targets_results"




def apply_random_erasing(image, target, img_h, img_w,erase_prob=0.3):

    if random.random() < erase_prob:  # Apply erasing with given probability
        # Randomly choose the region to erase
        erase_h = random.randint(int(0.02 * img_h), int(0.2 * img_h))
        erase_w = random.randint(int(0.02 * img_w), int(0.2 * img_w))
        i = random.randint(0, img_h - erase_h)
        j = random.randint(0, img_w - erase_w)

        # Erase region by setting pixel values to zero (black)
        image = TF.erase(image, i, j, erase_h, erase_w, v=0)
        target = TF.erase(target, i, j, erase_h, erase_w, v=0)  # Apply same erasing to target

    return image, target

def add_salt_and_pepper_noise(image, prob):
    """
    Adds Salt-and-Pepper noise to an image.
    
    """
    mask = torch.rand(image.shape)  # Generate random noise mask
    salt_mask = mask < (prob / 2)  # Assign salt (white) pixels
    pepper_mask = mask > (1 - prob / 2)  # Assign pepper (black) pixels

    noisy_image = image.clone()  # Copy the original image
    noisy_image[salt_mask] = 1.0  # Set salt pixels to white
    noisy_image[pepper_mask] = 0.0  # Set pepper pixels to black
    return noisy_image


def augment_images(input_folder, target_folder, output_folder_images, output_folder_targets, num_augmentations):
    """
    Applies identical data augmentations to images and their corresponding targets.

    """
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    for img_name in tqdm(image_files):
        img_path = os.path.join(input_folder, img_name)
        #target finishes in png
        target_name= img_name[:-3] + "png"
        target_path = os.path.join(target_folder, target_name)

        # Ensure the target exists
        if not os.path.exists(target_path):
            print(f"\nWarning: No target found for {img_name}, skipping...")
            continue
        
        # Read image and target
        img = read_image(img_path).float() / 255.0  # Normalize to [0,1]
        target = read_image(target_path).float() / 255.0  # Normalize target as well

        # Convert RGBA to RGB if needed
        if img.shape[0] == 4:
            img = img[:3, :, :]
        if target.shape[0] == 4:
            target = target[:3, :, :]

        iteration=0
        for iteration in range(num_augmentations):
            # Generate random transformation parameters
            angle = random.uniform(-30, 30)
            hflip = random.random() > 0.5
            yflip = random.random() > 0.5
            scale = random.uniform(0.8, 1.0)
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.8, 1.0), ratio=(0.75, 1.33))

            # ROTATION
            img_aug = TF.rotate(img, angle)
            target_aug = TF.rotate(target, angle)

            #HORIZONTAL FLIP
            if hflip:
                img_aug = TF.hflip(img_aug)
                target_aug = TF.hflip(target_aug)

            #VERTICAL FLIP
            if yflip:
                img_aug = TF.vflip(img_aug)
                target_aug = TF.vflip(target_aug)

            #TRANSLATION
            height= img.shape[1]
            width= img.shape[2]

            x = random.randint(-int(width/4), int(width/4))
            y = random.randint(-int(height/4), int(height/4))

            img_aug=TF.affine(img_aug, angle=0, translate=[x, y], scale=1, shear=0)
            target_aug=TF.affine(target_aug, angle=0, translate=[x, y], scale=1, shear=0)

            #RANDOM ERASING
            img_aug, target_aug = apply_random_erasing(img_aug, target_aug, height, width)


            #CROP
            img_aug = TF.resized_crop(img_aug, i, j, h, w, (img.shape[1], img.shape[2]))
            target_aug = TF.resized_crop(target_aug, i, j, h, w, (target.shape[1], target.shape[2]))

            # Calculate padding
            left = (img.shape[2] - img_aug.shape[2]) // 2
            right = img.shape[2] - img_aug.shape[2] - left
            top = (img.shape[1] - img_aug.shape[1]) // 2
            bottom = img.shape[1] - img_aug.shape[1] - top

            img_aug = TF.pad(img_aug, padding=[left, top, right, bottom], fill=0)
            target_aug = TF.pad(target_aug, padding=[left, top, right, bottom], fill=0)

            
            # Apply padding
            img_aug = TF.pad(img_aug, padding=[left, top, right, bottom], fill=0)
            target_aug = TF.pad(target_aug, padding=[left, top, right, bottom], fill=0)

            img_aug = TF.adjust_brightness(img_aug, brightness_factor=random.uniform(0.8, 1.2))
            img_aug = TF.adjust_contrast(img_aug, contrast_factor=random.uniform(0.8, 1.2))

            
            #NOISE INJECTION
            prob=random.uniform(0, 0.1)
            img_aug=add_salt_and_pepper_noise(img_aug, prob)


            # Save augmented images with the same filename
            save_image(img_aug, os.path.join(output_folder_images, f"aug_{iteration}_{img_name}"))
            save_image(target_aug, os.path.join(output_folder_targets, f"aug_{iteration}_{target_name}"))
    
    print(f"Augmentation complete! Augmented images saved in {output_folder_images} and {output_folder_targets}")



augment_images(input_folder, target_folder, output_folder_images, output_folder_targets,num_augmentations)
