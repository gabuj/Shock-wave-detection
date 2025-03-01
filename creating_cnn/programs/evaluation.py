import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from creating_cnn.deprecated_programs.training_dataset_old import ShockWaveDataset
import matplotlib.pyplot as plt
from torch import nn
from cnn_architecture import UNet

#adjustable parameters
batch_size = 1


# Define paths to your image and label directories
images_dir = "creating_training_set/schockwaves_images_used"
labels_dir = "creating_training_set/calibrated_training_images"

train_file_path = "creating_cnn/outputs/temporary/train_files.json"
test_file_path = "creating_cnn/outputs/temporary/test_files.json"



model_path = "creating_cnn/outputs/models/model.pth"
# Initialize the model (same architecture as during training)
model = UNet(pretrained=False)  # No need to load pretrained weights for this case
# Load the trained weights into the model
model.load_state_dict(torch.load(model_path))


#acquire data loaders

# Load the filenames from the JSON file
with open(train_file_path, 'r') as f:
    train_files = json.load(f)

with open(test_file_path, 'r') as f:
    test_files = json.load(f)


# Define the transformations (if any)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (0-1 range)
])

# Recreate the datasets
train_dataset = ShockWaveDataset(images_dir, labels_dir, train_files, transform=transform)
test_dataset = ShockWaveDataset(images_dir, labels_dir, test_files, transform=transform)

# Recreate the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Set the model to evaluation mode (disable dropout, batch normalization)
model.eval()

# Initialize metrics
total_loss = 0.0
iou_scores = []

criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification (shock wave vs. non-shock wave)


# Disable gradient calculation during evaluation
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.cuda()
        labels = labels.cuda()

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

        # Optionally visualize output
        binary_output = binary_output.squeeze().cpu().numpy()  # Remove batch and channel dimensions
        plt.imshow(binary_output, cmap='gray')
        plt.show()

# Print evaluation results
avg_loss = total_loss / len(test_dataloader)
avg_iou = sum(iou_scores) / len(iou_scores)

print(f"Test Loss: {avg_loss}")
print(f"Average IoU: {avg_iou}")

