import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import cv2
import os
from PIL import Image

# Simple CNN for edge detection
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Convolution layer
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Max pooling

    nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Convolution layer
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Max pooling

    nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Convolution layer
    nn.ReLU(),

    nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Output layer
    nn.Sigmoid()  # Sigmoid activation to get values between 0 and 1
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Image Transform (to tensor and resize)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to fixed size
    transforms.ToTensor(),
])

# Load data (shockwave images and corresponding edge maps)
image_paths = ['train_1'] # Update paths
edge_paths = ['label_1']  # Update paths


# Prepare data loader
def load_data(image_paths, edge_paths, transform):
    images = []
    edges = []

    for i in range(len(image_paths)):
        img = Image.open(image_paths[i]).convert('L')  # Grayscale
        edge = Image.open(edge_paths[i]).convert('L')  # Grayscale

        img = transform(img)
        edge = transform(edge)

        images.append(img)
        edges.append(edge)

    images = torch.stack(images)  # Stack all images into a batch
    edges = torch.stack(edges)  # Stack all edge images into a batch

    return images, edges


images, edges = load_data(image_paths, edge_paths, transform)

# Put model on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
images, edges = images.to(device), edges.to(device)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(images)

    # Compute loss
    loss = criterion(outputs, edges)

    # Backpropagation
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Save the model after training
torch.save(model.state_dict(), 'edge_detection_cnn.pth')
