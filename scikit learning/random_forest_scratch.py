import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = "/creating_training_set/schockwaves_images_isedairbos_f7_p5.png"  # Update path if needed
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not read the image. Check the file path!")
    exit()

# Apply Scharr filter in X and Y directions
scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)  # Horizontal edges
scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)  # Vertical edges

# Compute gradient magnitude
scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
scharr_magnitude = np.uint8(scharr_magnitude)  # Convert to uint8 for thresholding

# Apply Otsu's thresholding to create a binary image
_, scharr_edges = cv2.threshold(scharr_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(scharr_edges, cmap="gray")
plt.title("Scharr Edge Detection (Otsu Threshold)")
plt.axis("off")

plt.show()
