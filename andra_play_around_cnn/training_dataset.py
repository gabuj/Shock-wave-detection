import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ShockWaveDataset(Dataset):
    def __init__(self, images_dir, labels_dir, files, transform=None):
        """
        Args:
            images_dir (str): Path to the directory containing the images.
            labels_dir (str): Path to the directory containing the labels.
            files (list): List of filenames to use in the dataset.
            transform (callable, optional): Optional transform to be applied on an image and label pair.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.files = files  # List of image filenames
        self.transform = transform  # Image transformation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Get the filename
        image_file = self.files[idx]
        image_path = os.path.join(self.images_dir, image_file)

        # Get corresponding label filename (assuming labels have .png extension)
        label_file = os.path.splitext(image_file)[0] + ".png"
        label_path = os.path.join(self.labels_dir, label_file)

        # Load image (handle RGB and grayscale)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        if len(image.shape) == 3:  # Convert RGB to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Error loading label: {label_path}")

        # Ensure binary labels (0 or 1)
        label = (label > 127).astype('float32')

        # Convert to tensor and normalize
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize to [0,1]
        label = torch.tensor(label, dtype=torch.long)  # Binary mask

        # Apply transformations if provided
        #if self.transform:
        #    image = self.transform(image)
        #    label = self.transform(label)

        return image, label

# Example usage:
# dataset = ShockWaveDataset("path_to_images", "path_to_labels")
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
