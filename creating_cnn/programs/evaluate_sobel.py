import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from training_dataset import ShockWaveDataset
from torch import nn
from cnn_architecture_new import UNet
from useful_functions import evaluate
from useful_functions import load_filenames
from useful_functions import collate_fn
from useful_functions import compute_sobel
#adjustable parameters
batch_size = 1
threshold=10
edge_weight=1


# Define paths to your image and label directories
images_dir = "creating_training_set/schockwaves_images_used"
labels_dir = "creating_training_set/calibrated_training_images"

train_file_path = "creating_cnn/outputs/temporary/train_files.json"
test_file_path = "creating_cnn/outputs/temporary/test_files.json"


#GET DATA

# Define the transformations (if any)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (0-1 range)
])

#acquire filenames
train_files,test_files= load_filenames(train_file_path,test_file_path)

# Recreate the datasets
train_dataset = ShockWaveDataset(images_dir, labels_dir, train_files, transform=transform)
test_dataset = ShockWaveDataset(images_dir, labels_dir, test_files, transform=transform)

# Recreate the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) 


# Initialize metrics
total_loss = 0.0
iou_scores = []

weights = torch.tensor([edge_weight], dtype=torch.float32)

# Define the loss function with class weights
criterion = nn.BCELoss(weight=weights)


# Disable gradient calculation during evaluation
for inputs, labels in test_dataloader:
# Forward pass through the model
    sobel_image=compute_sobel(inputs)



    # Calculate loss
    loss = criterion(sobel_image, labels)
    total_loss += loss.item()

    # Visualize output and calculate iou
    iou_scores=evaluate(inputs, labels, sobel_image, iou_scores, threshold, show=1,compare=0)
    
# Print evaluation results
avg_loss = total_loss / len(test_dataloader)
avg_iou = sum(iou_scores) / len(iou_scores)

print(f"Test Loss: {avg_loss}")
print(f"Average IoU: {avg_iou}")