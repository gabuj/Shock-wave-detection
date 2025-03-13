import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from training_dataset import ShockWaveDataset
from torch import nn
from cnn_architecture_new import UNet
from useful_functions import evaluate
from useful_functions import load_filenames
from useful_functions import collate_fn
import os
#adjustable parameters
batch_size = 1
threshold=10
edge_weight=1


# Define paths to your image and label directories
dir = "creating_cnn/evaluation_images/random_forest"
labels_dir = "creating_training_set/calibrated_training_images"

#GET DATA

# Define the transformations (if any)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (0-1 range)
])
test_files = os.listdir(dir)

#acquire filenames
test_dataset = ShockWaveDataset(dir, labels_dir, test_files, transform=transform)

# Recreate the DataLoaders
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Initialize metrics
total_loss = 0.0
iou_scores = []

weights = torch.tensor([edge_weight], dtype=torch.float32)

# Define the loss function with class weights
criterion = nn.BCELoss(weight=weights)


# Disable gradient calculation during evaluation
with torch.no_grad():
    for inputs, labels in test_dataloader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Calculate loss
        loss = criterion(inputs, labels)
        total_loss += loss.item()

        # Visualize output and calculate iou
        iou_scores=evaluate(inputs, labels, inputs, iou_scores, threshold, show=0,compare=0)
        
# Print evaluation results
avg_loss = total_loss / len(test_dataloader)
avg_iou = sum(iou_scores) / len(iou_scores)

print(f"Test Loss: {avg_loss}")
print(f"Average IoU: {avg_iou}")