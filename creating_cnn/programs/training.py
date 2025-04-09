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



# Start time
start_time = time.time()
# Adjustable parameters
model_path = "creating_cnn/outputs/models/model_B.pth"
batch_size = 1
learning_rate = 1e-4
num_epochs = 5
test_size=0.2
threshold=10
edge_weight=8


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

train_dataloader, test_dataloader=create_dataloader(images_dir, labels_dir,train_file_path,test_file_path,transform, batch_size,test_size)
print("created dataloaders")

# # Initialize the U-Net model
# model = UNet()
# if torch.cuda.is_available():
#     model = model.cuda()  # Move the model to GPU if available, it's faster to train on GPU

#define device
if torch.backends.mps.is_available():
    device = "mps"  # Use Apple's Metal (MPS) acceleration
elif torch.cuda.is_available():
    device = "cuda"  # Use CUDA if available (for NVIDIA GPUs)
else:
    device = "cpu"  # Fallback to CPU

print(f"Using device: {device}")

#initialize model to device
model=UNet().to(device)


# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
#for criterion use combined weighted cross entropy loss
weights = torch.tensor([edge_weight], dtype=torch.float32)

# Define the loss function with class weights
criterion = nn.BCELoss(weight=weights)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

print("initialized model, optimizer and loss function, now starting training")
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
        elif torch.backends.mps.is_available():
            print("you must have a nice apple computer")
            inputs = inputs.mps()   # Move inputs to Apple's Metal (MPS) acceleration if available
            labels = labels.cuda()  # Move labels to Apple's Metal (MPS) acceleration if available
        
        optimizer.zero_grad()  # Zero the gradients before each step

        # Forward pass
        outputs = model(inputs)

        #add num classes to labels:

        # Compute loss
        loss = criterion(outputs, labels)# + dice_loss(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    

    avg_train_loss = running_loss / len(train_dataloader)
    #print loss
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.6f}")
    print("time since start: ", (time.time()-start_time)/60, " minutes")
    # Update the learning rate based on loss
    scheduler.step(avg_train_loss)
    print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")


# Save the trained model
torch.save(model.state_dict(), model_path)
print("model saved")


# MODEL EVALUATION
model.eval() # Set the model to evaluation mode

# Initialize metrics
total_loss = 0.0    #as before, total loss is the sum of the loss for each batch
iou_scores = []

print("starting evaluation")
with torch.no_grad(): # Disable gradient calculation during evaluation
    for inputs, labels in test_dataloader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Forward pass through the model
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

# Visualize output and calculate iou
        iou_scores=evaluate(inputs, labels, outputs, iou_scores, threshold, show=0,compare=0)
        
# Print evaluation results
avg_loss = total_loss / len(test_dataloader)
avg_iou = sum(iou_scores) / len(iou_scores)

print(f"Test Loss: {avg_loss}")
print(f"Average IoU: {avg_iou}")