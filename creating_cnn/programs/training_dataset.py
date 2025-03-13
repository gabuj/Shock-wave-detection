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
        self.files = files  # This should be a list of image filenames
        self.transform = transform  # Store the transform to be applied later
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Get the filename for this sample
        image_file = self.files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        #labels have same name but .png extension
        label_file = image_file[:-4]+".png"
        label_path = os.path.join(self.labels_dir, label_file)  # Assuming labels have the same name as images
        
        # Load image (handle RGB and grayscale)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:  # Convert RGB to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load label (always grayscale)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = (label > 127).astype('float32')  # Ensure binary labels

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return image, label


# Example usage:
# dataset = ShockWaveDataset("path_to_images", "path_to_labels")
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
