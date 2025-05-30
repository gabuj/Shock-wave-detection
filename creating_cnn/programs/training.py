import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from cnn_architecture_new import UNet
from useful_functions import combined_loss
# from sklearn.model_selection import train_test_split
from torchvision import transforms
# from useful_functions import evaluate
from useful_functions import create_dataloader
import time
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Start time
start_time = time.time()

# Adjustable parameters - OTTIMIZZATI
model_path = "creating_cnn/outputs/models/model_datasetTest.pth"
batch_size = 4  # Aumentato da 1 a 8 per stabilizzare l'addestramento
base_learning_rate = 3e-4  #  with cosine annealing, over 5e-3 it learns wrong! 5e-4 is still to high, down to 0.49... 1e-4 gets down to 0.45 in 10 epochs, 2e-4 gets down to 0.39 but can't get better after, 3e-4 down to .38 
max_lr=base_learning_rate * 2
min_learning_rate=base_learning_rate / 100  # Minimum learning rate after annealing

num_epochs = 20  # Number of epochs to train
test_size = 0.2

bce_weight = 0.5  # will have to discover
fp_weight=0.7 # 0.6 bad, 0.7 badish, 0.5 same,0.8 same 0.9 white, 1 worse, 1.2 white, 1.4 white, 1.6 black

gamma_focal=1.1 #1 good, 2 seems bad, 1.2 maybe
patience = int(num_epochs/7)  # Per early stopping

# Define paths to your image and label directories
images_dir = "creating_cnn/final_images/images"
labels_dir = "creating_cnn/final_images/targets"

train_file_path = "creating_cnn/outputs/temporary/train_files.json"
test_file_path = "creating_cnn/outputs/temporary/test_files.json"

# CREATE DATA LOADER
# Definisci trasformazioni avanzate per data augmentation
transform_train = transforms.Compose([
    transforms.ToTensor(),  # Conversione a tensore
    transforms.Normalize([0.5], [0.5])  # Normalizzazione per immagini a singolo canale
])

transform_test = transforms.Compose([
    transforms.ToTensor(),  # Conversione a tensore
    transforms.Normalize([0.5], [0.5])  # Normalizzazione per immagini a singolo canale
])

# Modifica la funzione create_dataloader per supportare trasformazioni separate e set di validazione
# Se la tua funzione create_dataloader non supporta questo, dovrai riscriverla o adattare questo codice
train_dataloader, test_dataloader = create_dataloader(
    images_dir, labels_dir, train_file_path, test_file_path, 
    transform_train, batch_size, test_size
)
print("Created dataloaders")

# Define device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple's Metal (MPS) acceleration
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA if available (for NVIDIA GPUs)
else:
    device = torch.device("cpu")  # Fallback to CPU

print(f"Using device: {device}")

# Initialize model to device
model = UNet().to(device)

# Improved optimizer with weight decay regularization
optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=1e-4)

## ===== OPTIMIZED LEARNING RATE SCHEDULER =====
# # Using OneCycleLR for dynamic learning rate adjustment
# scheduler = OneCycleLR(
#     optimizer,
#     max_lr=max_lr,  # Peak learning rate is 10x base
#     steps_per_epoch=len(train_dataloader),
#     epochs=num_epochs,
#     pct_start=0.3,  # 30% of training for warmup phase
#     anneal_strategy='cos',  # Cosine annealing for smooth decay
#     div_factor=25,  # Initial learning rate = max_lr/div_factor (lower than base)
#     final_div_factor=1000,  # Final learning rate = initial_lr/final_div_factor
# )


## ===== OPTIMIZED LEARNING RATE SCHEDULER - CosineAnnealingWarmRestarts =====
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,  # Number of iterations for the first restart (adjustable)
    T_mult=2,  # Factor by which the number of iterations is multiplied after each restart
    eta_min=min_learning_rate,  # Minimum learning rate after annealing
    last_epoch=-1
)

# Function to track learning rates
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Funzione di early stopping
def early_stopping(val_losses, patience=5, min_delta=0.01):
    #this function checks if the validation loss has stopped improving
    if len(val_losses) < patience + 1:
        return False
    
    recent_min = min(val_losses[-patience:])
    if val_losses[-patience-1] < recent_min + min_delta:
        return True
    return False

print("Initialized model, optimizer and scheduler, now starting training")

# Variabili per tracking
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_path = model_path.replace('.pth', '_best.pth')

# Training loop
model.train()  # Set the model to training mode

for epoch in range(num_epochs):
    model.train()  # Set to training mode for each epoch
    running_loss = 0.0
    batch_lr = []  # Track LR for each batch

    # Loop through the training dataloader
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Track learning rate
        batch_lr.append(get_lr(optimizer))
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss usando la funzione di loss combinata
        loss = combined_loss(outputs, labels, bce_weight=bce_weight, fp_weight=fp_weight, gamma=gamma_focal)

        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Step the scheduler after each batch (OneCycleLR is designed to update per batch)
        scheduler.step()

        running_loss += loss.item()
    
    # Calcola la loss media per questa epoca
    avg_train_loss = running_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    
    # Validazione
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_dataloader:  # Usa il test dataloader come validazione
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = combined_loss(outputs, labels, bce_weight=bce_weight, fp_weight=fp_weight, gamma=gamma_focal)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(test_dataloader)
    val_losses.append(avg_val_loss)

    # Stampa le statistiche
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Time since start: {(time.time()-start_time)/60:.2f} minutes")
    
    # Salva il miglior modello
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model with validation loss: {best_val_loss:.6f}")
    
    # Early stopping
    if early_stopping(val_losses, patience=patience, min_delta=0.001):
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

# Salva il modello finale
torch.save(model.state_dict(), model_path)
print("Final model saved")

# Visualizza curve di loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('creating_cnn/outputs/loss_curves.png')
print("Loss curves saved to 'creating_cnn/outputs/loss_curves.png'")