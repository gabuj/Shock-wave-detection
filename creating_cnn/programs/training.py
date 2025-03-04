import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn_architecture import UNet
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import json
from training_dataset import ShockWaveDataset

# Adjustable parameters
model_path = "creating_cnn/outputs/models/model.pth"
batch_size = 2
learning_rate = 1e-4
num_epochs = 10

# Define paths to your image and label directories
images_dir = "creating_training_set/schockwaves_images_used"
labels_dir = "creating_training_set/calibrated_training_images"

train_file_path = "creating_cnn/outputs/temporary/train_files.json"
test_file_path = "creating_cnn/outputs/temporary/test_files.json"

# CREATE DATA LOADER

# Define the transformations (if any)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (0-1 range)
])

image_files = os.listdir(images_dir)  # Path to your images directory

# Split into train and test sets (80% train, 20% test)
train_files, test_files = train_test_split(image_files, test_size=0.1, random_state=42)

# Create datasets and dataloaders for both train and test
train_dataset = ShockWaveDataset(images_dir, labels_dir, train_files, transform=transform)
test_dataset = ShockWaveDataset(images_dir, labels_dir, test_files, transform=transform)
    

# Save filenames to JSON files so they can be used later
train_files = list(train_dataset.files)
test_files = list(test_dataset.files)

with open(train_file_path, 'w') as f:
    json.dump(train_files, f)

with open(test_file_path, 'w') as f:
    json.dump(test_files, f)

# Create DataLoader for both training and testing
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Training set should shuffle
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Test set should NOT shuffle

# Initialize the U-Net model
model = UNet()
if torch.cuda.is_available():
    model = model.cuda()  # Move the model to GPU if available, it's faster to train on GPU

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification (shock wave vs. non-shock wave)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  #running loss is the loss for the current epoch

    # Loop through the dataloader
    for inputs, labels in train_dataloader:
        if torch.cuda.is_available():
            print("you must have a nice GPU")
            inputs = inputs.cuda()  # Move inputs to GPU if available
            labels = labels.cuda()  # Move labels to GPU if available

        optimizer.zero_grad()  # Zero the gradients before each step

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print loss for every epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}")

# Save the trained model
torch.save(model.state_dict(), model_path)

# MODEL EVALUATION
# Set the model to evaluation mode (disable dropout, batch normalization)
model.eval()

# Initialize metrics
total_loss = 0.0    #as before, total loss is the sum of the loss for each batch
iou_scores = []

# Disable gradient calculation during evaluation
with torch.no_grad():
    for inputs, labels in test_dataloader:
        if torch.cuda.is_available():
            print("again, nice GPU")
            inputs = inputs.cuda()  # Move inputs to GPU if available
            labels = labels.cuda()  # Move labels to GPU if available

        # Forward pass through the model
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Post-process outputs (convert to binary masks)
        binary_output = (outputs > 0.5).float() * 255

        # Calculate Intersection over Union (IoU)
        intersection = torch.sum(binary_output * labels)
        union = torch.sum(binary_output) + torch.sum(labels) - intersection
        iou = intersection / union if union != 0 else 0
        iou_scores.append(iou.item())

        # Optionally visualize output together with the ground truth
        binary_output = binary_output.squeeze().cpu().numpy()
        plt.imshow(binary_output, cmap='gray')
        plt.imshow(labels.squeeze().cpu().numpy(), cmap='gray', alpha=0.5) #not sure it will work, trying to overlay the ground truth
        plt.show()
        
# Print evaluation results
avg_loss = total_loss / len(test_dataloader)
avg_iou = sum(iou_scores) / len(iou_scores)

print(f"Test Loss: {avg_loss}")
print(f"Average IoU: {avg_iou}")