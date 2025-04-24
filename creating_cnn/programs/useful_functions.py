import torch
import torch.nn.functional as F
from training_dataset import ShockWaveDataset
from sklearn.model_selection import train_test_split
import json
import os
from torch.utils.data import DataLoader
from skimage import filters
import numpy as np
from skimage.filters import threshold_otsu


def collate_fn(batch):
    """
    Custom collate function to pad images and labels in a batch to the same size.
    """
    images, masks = zip(*batch)  # Unzip images and corresponding masks

    # Find max height and width in the batch
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    # Pad all images to the same size
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in images]

    # Pad masks: The labels should have shape [batch_size, height, width] with class indices
    padded_masks = [F.pad(mask, (0, max_width - mask.shape[2], 0, max_height - mask.shape[1])) for mask in masks]

    # Stack to create batch
    # Convert images and masks into a tensor (ensure they are torch tensors)
    padded_images = torch.stack(padded_images)  # Shape: [batch_size, 1, H, W] for grayscale images
    padded_masks = torch.stack(padded_masks)  # Shape: [batch_size, H, W] for class indices

    return padded_images, padded_masks

#show image:
import cv2
import matplotlib.pyplot as plt

def write_json_file(train_file_path,test_file_path, train_files,test_files):
    with open(train_file_path, 'w') as f:
        json.dump(train_files, f)

    with open(test_file_path, 'w') as f:
        json.dump(test_files, f)




#def augmentation()# use ImageDataGenerator?

def create_dataloader(images_dir, labels_dir,train_file_path,test_file_path,transform, batch_size,test_size):
    # Split into train and test sets (80% train, 20% test)
    print("Looking for images in:", images_dir)
    print("Absolute path:", os.path.abspath(images_dir))

    image_files = os.listdir(images_dir)  # Path to your images directory


    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)

    # Create datasets and dataloaders for both train and test
    train_dataset = ShockWaveDataset(images_dir, labels_dir, train_files, transform=transform)
    test_dataset = ShockWaveDataset(images_dir, labels_dir, test_files, transform=transform)
    print("acquired datasets")

    # Save filenames to JSON files so they can be used later
    train_files = list(train_dataset.files)
    test_files = list(test_dataset.files)

    #write json files
    write_json_file(train_file_path,test_file_path, train_files,test_files)
    print("saved training and testing filenames")

    # Create DataLoader for both training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  # Test set should NOT shuffle
    return train_dataloader, test_dataloader

def load_filenames(train_file_path,test_file_path):
    # Load the filenames from the JSON file
    with open(train_file_path, 'r') as f:
        train_files = json.load(f)

    with open(test_file_path, 'r') as f:
        test_files = json.load(f)
    return train_files,test_files

def compare_outputs(input, label, binary_output):
    #get each image, label, and output from the batch
    plt.subplot(1,3,1)
    plt.imshow(binary_output, cmap='gray')
    plt.title("Predicted Mask")
    plt.subplot(1,3,2)
    plt.imshow(label, cmap='gray')
    plt.title("Ground Truth")
    plt.subplot(1,3,3)
    plt.imshow(input, cmap='gray')
    plt.title("Input Image")
    plt.show()
    return binary_output

def show_output(binary_output, threshold):
    plt.imshow(binary_output, cmap='gray')
    plt.title("Predicted Mask")
    plt.show()
    return binary_output
    
def evaluate(inputs, labels, outputs, iou_scores, threshold, show=0,compare=bool(0)):
    temp_iou=[]
    for i in range(inputs.size(0)): 
        input = inputs[i].squeeze().cpu().numpy()
        label = labels[i].squeeze().cpu().numpy()

        # # Convert logits to probabilities using softmax
        # output_probs = F.softmax(outputs[i], dim=1)  # Shape: [1, 2, H, W]
        # # Select the most probable class (argmax over the 2 channels)
        # predicted_mask = torch.argmax(output_probs, dim=0)  # Shape: [1, H, W]
        # output = predicted_mask.cpu().numpy()

        output = outputs[i].squeeze().cpu().numpy()

        output = output - output.min()
        output = output / output.max()
        output= output*255
        binary_output = (output > threshold)            
        if show != 0:
            show_output(binary_output, threshold)
        if compare != 0:
            compare_outputs(input, label, binary_output)
        # Calculate Intersection over Union (IoU) for each image in the batch
        intersection = (binary_output * label).sum()
        union = binary_output.sum() + label.sum() - intersection
        iou = intersection / union
        temp_iou.append(iou)
    iou_scores.extend(temp_iou)
    return iou_scores

def dice_loss(pred, target, smooth=1.):
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice



def compute_sobel(input_tensor):
    """
    Computes the Sobel edge detection on a grayscale input image tensor.
    Args:
        input_tensor: Tensor of shape [batch, 1, height, width] with values in [0,1].
    Returns:
        sobel_image: Tensor of the same shape containing Sobel-filtered edges.
    """
    # Ensure input is float and has 4D shape [batch, 1, H, W]
    if input_tensor.ndim == 3:  # If shape is [1, H, W], add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
    
    input = input_tensor.squeeze().cpu().numpy()
    sobel_image=filters.sobel(input)


    # Normalize to [0,1] for visualization
    sobel_image = (sobel_image - sobel_image.min()) / (sobel_image.max() - sobel_image.min())

    #reconvert to tensor
    sobel_image = torch.tensor(sobel_image, dtype=torch.float32).unsqueeze(0)
    sobel_image = sobel_image.unsqueeze(0)
    return sobel_image

def visualize_single_image(input_tensor, output_tensor, original_image, threshold=0.5, use_otsu=False):
    output_np = output_tensor.squeeze().cpu().numpy()

    # Normalize output to [0, 255]
    output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min() + 1e-8)
    output_np = (output_np * 255).astype(np.uint8)

    if use_otsu:
        threshold = threshold_otsu(output_np)
        print(f"Computed Otsu threshold: {threshold}")

    binary_output = (output_np > threshold).astype(np.uint8)

    # --- Display side by side ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(output_np, cmap='gray')
    axs[1].set_title("Raw Model Output")
    axs[1].axis('off')

    axs[2].imshow(binary_output, cmap='gray')
    axs[2].set_title(f"Binarized Output (threshold={threshold:.3f})")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


def combined_loss(pred, target, bce_weight=0.5, fp_weight=2.0, gamma=2.0):
    # Create weight tensor for BCE to penalize false positives more
    # Higher values where target=0 (non-shock pixels) to reduce overtracing
    weights = torch.ones_like(target)
    weights = weights + (fp_weight - 1.0) * (target)  # More weight where target=1 (shockwave pixels)

    # Focal loss component - focuses more on hard examples
    pt = target * pred + (1 - target) * (1 - pred)  # Probability of true class
    focal_weights = (1 - pt) ** gamma  # Focal loss factor, if very certain, weight is low, if uncertain, weight is high

    # Combine focal weights with our custom weights
    final_weights = weights * focal_weights

    # Weighted BCE using the built-in weight parameter
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    weighted_bce = (bce_loss * final_weights).mean()

    # Modified Dice Loss with emphasis on false negatives
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Standard intersection
    intersection = (pred_flat * target_flat).sum()

    # False positives (pred=1 where target=0)
    false_negatives = ((1 - pred_flat) * target_flat).sum()

    # Penalize false positives in denominator
    union = pred_flat.sum() + target_flat.sum() + (fp_weight - 1.0) * false_negatives

    # Modified Dice loss
    dice = 1 - (2.0 * intersection + smooth) / (union + smooth)  #

    # Combined loss
    return bce_weight * weighted_bce + (1 - bce_weight) * dice
