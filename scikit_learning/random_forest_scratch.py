import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from skimage import filters, color

# Define paths
trace_dir = "creating_training_set/shockwaves_images/"
trace_files = os.listdir(trace_dir)

threshold = 0.022
sample_size = 5  # Number of patches to visualize per image
patch_size = 15  # Change this to 5, 9, 15, etc.

features = []
labels = []

for image_file in trace_files:
    image_path = os.path.join(trace_dir, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {image_file} (could not read)")
        continue

    # Convert to grayscale and apply Sobel filter
    image = color.rgb2gray(image)
    sobel_image = filters.sobel(image)

    # Threshold edges
    binary_edges = (sobel_image > threshold).astype(np.uint8)

    height, width = binary_edges.shape
    fig, axes = plt.subplots(1, sample_size, figsize=(15, 3))  # Create figure for patches

    offset = patch_size // 2  # Half the patch size to ensure centered selection

    for i in range(sample_size):
        x, y = random.randint(offset, width - offset - 1), random.randint(offset, height - offset - 1)
        patch = image[y - offset:y + offset + 1, x - offset:x + offset + 1]  # Extract NxN patch

        if patch.shape == (patch_size, patch_size):  # Ensure correct patch size
            features.append(patch.flatten())
            labels.append(binary_edges[y, x])  # 1 for edge, 0 for background

            # Visualize the patch
            axes[i].imshow(patch, cmap="gray")
            axes[i].set_title(f"Label: {binary_edges[y, x]}")
            axes[i].axis("off")

    plt.suptitle(f"Patches from {image_file} (Patch Size: {patch_size}×{patch_size})")
    plt.show()

X = np.array(features)
y = np.array(labels)

print(f"Dataset size: {X.shape[0]} samples, each with {X.shape[1]} features.")
