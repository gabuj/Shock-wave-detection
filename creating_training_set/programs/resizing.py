import os
from PIL import Image


def resize_images(input_dir, output_dir, resize_factor=0.2):  # 0.2 means 20% of original size
    # Ensure the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)

        # Check if the file is an image (optional: you can expand the condition for different image formats)
        if image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            try:
                # Open the image
                with Image.open(image_path) as img:
                    # Calculate the new size (20% of original size)
                    width, height = img.size
                    new_size = (int(width * resize_factor), int(height * resize_factor))

                    # Resize the image using the LANCZOS resampling filter
                    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

                    # Save the resized image to the output directory
                    output_path = os.path.join(output_dir, image_name)
                    resized_img.save(output_path)

                    print(f"Resized and saved: {image_name}")
            except Exception as e:
                print(f"Error resizing image {image_name}: {e}")
# Example usage
input_directory = 'creating_training_set/calibrated_training_images'  # Input directory containing original images
output_directory = 'creating_training_set/calibrated_training_images_resized'  # Output directory to save resized images

resize_images(input_directory, output_directory)
