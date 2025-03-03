from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread("/shockwaves_images/train_1.png", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (5,5), 0)

# Compute Sobel gradients
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction

# Compute Edge Magnitude
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Normalize to 0-255
sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

# Display results
plt.figure(figsize=(12,4))
plt.subplot(1,3,1), plt.imshow(sobel_x, cmap='gray'), plt.title("Sobel X")
plt.subplot(1,3,2), plt.imshow(sobel_y, cmap='gray'), plt.title("Sobel Y")
plt.subplot(1,3,3), plt.imshow(sobel_magnitude, cmap='gray'), plt.title("Edge Magnitude")
plt.show()

'''
# Extract HOG features
hog_features, _ = hog(enhanced, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# Generate training data (label edges from Canny)
labels = (edges > 0).astype(int).flatten()
X_train = hog_features.reshape(-1, 1)
y_train = labels

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict edges
pred_edges = clf.predict(X_train).reshape(image.shape)

plt.imshow(pred_edges, cmap='gray')
plt.title("AI-Based Edge Detection (Random Forest)")
plt.show()
'''