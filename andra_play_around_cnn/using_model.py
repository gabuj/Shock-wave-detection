import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from cnn_architecture import UNet
from skimage import color

#image path

image_dir = "creating_training_set/schockwaves_images_used"
image_name = "Blunt_body_reentry_shapes1.png"
image_path = image_dir + "/" + image_name

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)




#acquire model

model_path = "creating_cnn/outputs/models/model.pth"
# Initialize the model (same architecture as during training)
model = UNet(pretrained=False)  # No need to load pretrained weights for this case
# Load the trained weights into the model
model.load_state_dict(torch.load(model_path))

# Define the transformations (if any)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (0-1 range)
])

#to use model, set to evaluation mode
model.eval()


#make image loadable 
if len(image.shape) == 3:  # Convert RGB to grayscale
    loadable_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to tensor
loadable_image = torch.tensor(loadable_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
# Now loadable_image has shape: (1, 1, height, width) (batch size, channels, height, width)

# Create output
output = model(loadable_image)

binary_output = (output > 0.5).float() * 255

# visualize output
binary_output = binary_output.squeeze().cpu().numpy()
plt.imshow(binary_output, cmap='gray')
plt.show()



#now change image to greyscale to compare
#make image grayscale and normalise
image = image[:,:,:3]
image = color.rgb2gray(image)

#normalise the image
image=image/image.max()
image=(image*255).astype('uint8')

#normalise output and convert to binary
binary_output = binary_output/binary_output.max()
binary_output = (binary_output * 255).astype('uint8')  # Convert the output to uint8


#convert binary output to 

# Compare images by blending them
alpha = 0.5
blended = cv2.addWeighted(image, alpha, binary_output, 1 - alpha, 0)

# Show images
plt.imshow(blended, cmap='gray')
plt.title("Blended Image for comparison")
plt.show()


#save output
output_path = "creating_cnn/outputs/predictions/first_try" + image_name
cv2.imwrite(output_path, binary_output)