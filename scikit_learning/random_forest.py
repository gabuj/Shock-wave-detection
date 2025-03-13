import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss
# Directories
IMAGE_DIR = "creating_training_set/schockwaves_images_used"
LABEL_DIR = "creating_training_set/calibrated_training_images"


results_images = ["f4_p3_cam_plane_drop_new_2-22-19.jpg",
                  "cak_colormap_0.jpg", "2005125111210_846.jpg",
                  "Screenshot from 2025-03-03 10-54-31.png",
                  "weird_fromarticle3.png", "Screenshot from 2025-03-03 10-53-39.png",
                  "airbos_f7_p5.jpg", "Screenshot from 2025-03-03 10-53-45.png"]
# Image settings
IMG_WIDTH, IMG_HEIGHT = 400, 400


def load_images(image_dir, label_dir, resize_shape=(400, 400)):
    """Loads images and their corresponding labels, preprocesses them, and returns feature/label arrays."""
    image_files = os.listdir(image_dir)
    X, y = [], []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".png")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if img is None or label is None:
            print(f"Skipping {img_file} due to loading error.")
            continue

        # Resize both images
        img = cv2.resize(img, resize_shape)
        label = cv2.resize(label, resize_shape)

        # Flatten and append
        X.extend(img.flatten().reshape(-1, 1))  # Pixel intensity as feature
        y.extend(label.flatten())  # Corresponding labels

    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Convert labels to binary (0 or 1)
    y = (y > 127).astype(int)

    return X, y

# Load dataset
X, y = load_images(IMAGE_DIR, LABEL_DIR, resize_shape=(IMG_WIDTH, IMG_HEIGHT))
print(np.unique(y))

print(f"Dataset size: {X.shape[0]} pixels")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# Train model
class_weights = {0: 1, 1: 18}
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights, criterion = 'entropy')

rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("Class weights:", class_weight_dict)

'''balanced weights gives
Model Accuracy: 0.4886
Class weights: {0: 0.5126727374797329, 1: 20.227387267338067}'''

def apply_sobel_filter(img, resize_shape=(400, 400)):
    """Apply Sobel filter to the unseen image (edge detection)."""
    img = cv2.resize(img, resize_shape)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)

    return sobel_edges

def test_unseen_image_sobel(image_path, model, resize_shape=(400, 400)):
    """Predicts pixel-wise classification for an unseen image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error loading image: {image_path}")
        return

    #

    sobel_img = apply_sobel_filter(img, resize_shape)
    img_flat = sobel_img.flatten().reshape(-1, 1)

    #img = cv2.resize(img, resize_shape)
    #img_flat = img.flatten().reshape(-1, 1)  # Prepare for prediction


    pred = model.predict(img_flat)  # Predict per pixel
    pred_img = pred.reshape(resize_shape) * 255  # Convert back to image format

    # Save output
    output_path = "scikit_learning/output_images/output_1_sobel.png"

    pred_img = pred_img.astype(np.uint8)

    cv2.imwrite(output_path, pred_img)
    print(f"Prediction saved to {output_path}")


def test_unseen_image(image_path, model, resize_shape=(400, 400)):
    """Predicts pixel-wise classification for an unseen image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error loading image: {image_path}")
        return

    #

    #sobel_img = apply_sobel_filter(img, resize_shape)
    #img_flat = sobel_img.flatten().reshape(-1, 1)

    img = cv2.resize(img, resize_shape)
    img_flat = img.flatten().reshape(-1, 1)  # Prepare for prediction

    pred = model.predict(img_flat)  # Predict per pixel
    pred_img = pred.reshape(resize_shape) * 255  # Convert back to image format

    # Save output
    output_path = "scikit_learning/output_images/output_1.png"

    pred_img = pred_img.astype(np.uint8)

    cv2.imwrite(output_path, pred_img)
    print(f"Prediction saved to {output_path}")

# Test on an unseen image
test_unseen_image("creating_training_set/shockwaves_images/Harold-E-Edgerton-Bullet-Shock-Wave.jpg", rf)

from sklearn.metrics import log_loss
import numpy as np


def compute_weighted_log_loss(y_true, y_prob, edge_weight=10.0):
    """Computes weighted log loss (cross-entropy) using scikit-learn."""

    # Assign weights: Edge pixels (y_true == 1) get weight 10, non-edge pixels (y_true == 0) get weight 1
    weights = np.where(y_true == 1, edge_weight, 1.0)

    # Compute weighted log loss
    loss = log_loss(y_true, y_prob, sample_weight=weights)

    return loss


# Get predicted probabilities from the Random Forest
y_pred = rf.predict_proba(X_test)[:, 1]  # Extract probability of class 1 (edge)

# Compute weighted log loss
weighted_log_loss = compute_weighted_log_loss(y_test, y_pred)

print(f"Weighted Log Loss: {weighted_log_loss:.4f}")

