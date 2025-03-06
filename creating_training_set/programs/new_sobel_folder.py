import os
import numpy as np
from skimage import io, filters
from skimage import img_as_ubyte
from skimage import color

# Directory paths
input_folder = '/Users/andracriscov/Documents/project Y3/repo/creating_training_set/shockwaves_images'
output_folder = '/Users/andracriscov/Documents/project Y3/repo/creating_training_set/sobel_filtered_images'

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define threshold value
threshold = 0.001  # Adjust this value based on your needs

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adjust based on your file types
        # Read image
        image_path = os.path.join(input_folder, filename)
        image = io.imread(image_path)

        # Check if the image has more than one channel (RGB), otherwise, it is grayscale
        if image.ndim == 3:  # RGB image
            image = image[:, :, :3]  # Remove alpha channel if it exists
            image = color.rgb2gray(image)  # Convert to grayscale
        elif image.ndim == 2:  # Grayscale image
            image = image  # Already grayscale, no need to convert

        # Apply Sobel filter to get edges
        sobel_image = filters.sobel(image)

        # Apply threshold to Sobel image
        sobel_image[sobel_image < threshold] = 0

        # Convert sobel_image to 3-channel (RGB) image
        sobel_image = np.stack([sobel_image] * 3, axis=-1)

        # Convert the floating point image to uint8 for saving
        sobel_image = img_as_ubyte(sobel_image)

        # Save the processed image to the output folder
        output_image_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_sobel.png')
        io.imsave(output_image_path, sobel_image)
        print(f"Processed and saved: {output_image_path}")

input_folder = '/Users/andracriscov/Documents/project Y3/repo/creating_training_set/shockwaves_images/simulated_images'
output_folder = '/Users/andracriscov/Documents/project Y3/repo/creating_training_set/sobel_filtered_images/simulated_images'

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define threshold value
threshold = 0.001  # Adjust this value based on your needs

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adjust based on your file types
        # Read image
        image_path = os.path.join(input_folder, filename)
        image = io.imread(image_path)

        # Check if the image has more than one channel (RGB), otherwise, it is grayscale
        if image.ndim == 3:  # RGB image
            image = image[:, :, :3]  # Remove alpha channel if it exists
            image = color.rgb2gray(image)  # Convert to grayscale
        elif image.ndim == 2:  # Grayscale image
            image = image  # Already grayscale, no need to convert

        # Apply Sobel filter to get edges
        sobel_image = filters.sobel(image)

        # Apply threshold to Sobel image
        sobel_image[sobel_image < threshold] = 0

        # Convert sobel_image to 3-channel (RGB) image
        sobel_image = np.stack([sobel_image] * 3, axis=-1)

        # Convert the floating point image to uint8 for saving
        sobel_image = img_as_ubyte(sobel_image)

        # Save the processed image to the output folder
        output_image_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_sobel.png')
        io.imsave(output_image_path, sobel_image)
        print(f"Processed and saved: {output_image_path}")
