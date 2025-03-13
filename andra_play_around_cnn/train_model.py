import torch
import torch.optim as optim
from useful_functions import DataLoader
from cnn_architecture_new import UNet  # Import your model architecture
from training_dataset import ShockWaveDataset  # Import dataset class
from weighted_cross_entropy_loss import WeightedCrossEntropyLoss  # Import custom loss function
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from cnn_architecture_new import UNet
from sklearn.model_selection import train_test_split
from torchvision import transforms
from useful_functions import evaluate
from useful_functions import create_dataloader
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from useful_functions import dice_loss

# Define paths
images_dir = "creating_training_set/schockwaves_images_used"
labels_dir = "creating_training_set/calibrated_training_images"

train_file_path = "creating_cnn/outputs/temporary/train_files.json"
test_file_path = "creating_cnn/outputs/temporary/test_files.json"


files = sorted(os.listdir(images_dir))  # Assuming filenames match

model_path = "andra_play_around_cnn/outputs/models/model.pth"
# Hyperparameters
batch_size = 4
learning_rate = 1e-4
num_epochs = 10

test_size=0.2
threshold=10

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (0-1 range)
])

train_dataloader, test_dataloader=create_dataloader(images_dir, labels_dir,train_file_path,test_file_path,transform, batch_size,test_size)
print("created dataloaders")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and dataloader
train_dataset = ShockWaveDataset(images_dir, labels_dir, files)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = UNet().to(device)
criterion = WeightedCrossEntropyLoss(weight_edge=10, weight_bg=1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)  # Raw logits
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

    # Save model every few epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"andra_play_around_cnn/outputs/models/unet_epoch_{epoch+1}.pth")

# Final model save
torch.save(model.state_dict(), "andra_play_around_cnn/outputs/models/unet_final.pth")
print("Training complete. Model saved.")
