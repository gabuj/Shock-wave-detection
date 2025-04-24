import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from cnn_architecture_new import UNet  # Replace or paste your model here
import torch.nn.functional as F
from useful_functions import visualize_single_image
import numpy as np
from skimage.filters import threshold_otsu



# ----- Configuration -----
model_path = "creating_cnn/outputs/models/model_A_i_best.pth"
image_path = 'creating_training_set/schockwaves_images_used/Blunt_body_reentry_shapes1.png'
#image_path = "creating_training_set/schockwaves_images_used/f4_p3_cam_plane_drop_new_2-22-19.jpg"
output_path = "creating_cnn/outputs/sample_prediction.png"
threshold = 200
use_otsu = False    # << Set this to True or False depending on what you want

# ----- Set device -----
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- Load model -----
model = UNet(pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----- Load and preprocess image -----
original_image = Image.open(image_path).convert('L')  # Grayscale
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
input_tensor = transform(original_image).unsqueeze(0).to(device)

# ----- Run inference -----
with torch.no_grad():
    output = model(input_tensor)
    output = F.interpolate(output, size=original_image.size[::-1], mode='bilinear', align_corners=False)

# ----- Visualize results with raw + binary prediction -----
visualize_single_image(input_tensor, output, original_image, threshold=threshold, use_otsu=use_otsu)

# ----- Save the binary prediction (optional) -----
binary_output = (output.squeeze().cpu().numpy() > (threshold_otsu(output.squeeze().cpu().numpy()) if use_otsu else threshold)).astype(np.uint8)
plt.imsave(output_path, binary_output, cmap='gray')
print(f"Prediction saved to: {output_path}")