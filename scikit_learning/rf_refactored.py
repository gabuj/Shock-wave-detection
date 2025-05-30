import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
import joblib  # To save the model

# Directories
IMAGE_DIR = "creating_training_set/schockwaves_images_used"
LABEL_DIR = "creating_training_set/calibrated_training_images"

RESULT_IMAGES = ["f4_p3_cam_plane_drop_new_2-22-19.jpg",
                  "cak_colormap_0.jpg", "2005125111210_846.jpg",
                  "Screenshot from 2025-03-03 10-54-31.png",
                  "weird_fromarticle3.png", "Screenshot from 2025-03-03 10-53-39.png",
                  "airbos_f7_p5.jpg", "Screenshot from 2025-03-03 10-53-45.png"]

# Image settings
IMG_WIDTH, IMG_HEIGHT = 600, 600

# Create an output folder for the predictions if it doesn't exist
OUTPUT_FOLDER = "scikit_learning/output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_images(image_dir, label_dir, result_images, resize_shape=(600, 600)):
    """Loads images and their corresponding labels, preprocesses them, and returns feature/label arrays."""
    image_files = os.listdir(image_dir)
    X, y = [], []

    for img_file in image_files:
        if img_file in result_images:  # Skip result images
            continue

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

def train_model(X_train, y_train, class_weights):
    """Trains a RandomForest model with the specified class weights."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights, criterion='entropy', max_depth = 6)
    rf.fit(X_train, y_train)
    return rf

def save_model(model, edge_weight, filename="/Users/andracriscov/Documents/project Y3/repo/scikit_learning/model"):
    """Saves the trained model to a file with the edge weight in the filename."""
    model_filename = f"{filename}_edge_weight_{edge_weight}.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

def test_unseen_image(image_path, model, resize_shape=(600, 600)):
    """Predicts pixel-wise classification for an unseen image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error loading image: {image_path}")
        return

    img_resized = cv2.resize(img, resize_shape)

    original_output_path = os.path.join(OUTPUT_FOLDER,
                                        os.path.splitext(os.path.basename(image_path))[0] + "_original.png")
    cv2.imwrite(original_output_path, img_resized)
    print(f"Original resized image saved to {original_output_path}")

    # Resize image and flatten for prediction

    img_flat = img_resized.flatten().reshape(-1, 1)  # Flatten into a single sample

    pred = model.predict(img_flat)  # Predict per pixel
    pred_img = pred.reshape(resize_shape) * 255

    # Predict probabilities for class 1 (edge class)
    #y_prob = model.predict_proba(img_flat)[:, 1]  # Get the probability of class 1 (edge)

    # Reshape the prediction to match the image shape

    # Save output
    output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(os.path.basename(image_path))[0] + ".png")
    pred_img = pred_img.astype(np.uint8)

    cv2.imwrite(output_path, pred_img)
    print(f"Prediction saved to {output_path}")

    return pred_img

def test_unseen_image_2(image_path, model, resize_shape=(600, 600)):
    """Predicts pixel-wise classification for an unseen image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # Save original image (without resizing) to the output folder
    original_output_path = os.path.join(OUTPUT_FOLDER,
                                        os.path.splitext(os.path.basename(image_path))[0] + "_original.png")
    cv2.imwrite(original_output_path, img)
    print(f"Original image saved to {original_output_path}")

    # Resize the image to the required input size for prediction
    img_resized = cv2.resize(img, resize_shape)

    # Flatten and predict
    img_flat = img_resized.flatten().reshape(-1, 1)  # Flatten into a single sample
    pred = model.predict(img_flat)  # Predict per pixel
    pred_img = pred.reshape(resize_shape) * 255  # Rescale to 0-255 range for image saving
    pred_img = pred_img.astype(np.uint8)

    # Save the predicted image (in resized form)
    #resized_output_path = os.path.join(OUTPUT_FOLDER,
    #                                   os.path.splitext(os.path.basename(image_path))[0] + "_pred_resized.png")

    #cv2.imwrite(resized_output_path, pred_img)
    #print(f"Prediction saved (resized) to {resized_output_path}")

    # Save the predicted image in the original image size
    print('line 141', pred_img.shape, img.shape[1], img.shape[0])

    pred_img_original_size = cv2.resize(pred_img, (img.shape[1], img.shape[0]))

    original_size_output_path = os.path.join(OUTPUT_FOLDER,
                                             os.path.splitext(os.path.basename(image_path))[0] + "_pred_original.png")
    cv2.imwrite(original_size_output_path, pred_img_original_size)
    print(f"Prediction saved (original size) to {original_size_output_path}")

    return pred_img


def compute_weighted_log_loss(y_true, y_prob, edge_weight=10.0):
    """Computes weighted log loss (cross-entropy) using scikit-learn."""
    weights = np.where(y_true == 1, edge_weight, 1.0)
    loss = log_loss(y_true, y_prob, sample_weight=weights)
    return loss


def compute_iou(y_true, y_pred, threshold=0.5):
    """
    Compute IoU (Intersection over Union) for binary classification.
    """
    y_pred_binary = (y_pred > threshold).astype(int)  # Threshold predictions
    intersection = (y_true & y_pred_binary).sum()
    union = (y_true | y_pred_binary).sum()

    if union == 0:
        return 1.0  # Perfect IoU case (both empty)

    return intersection / union

# Load dataset and exclude result images
X, y = load_images(IMAGE_DIR, LABEL_DIR, RESULT_IMAGES, resize_shape=(IMG_WIDTH, IMG_HEIGHT))
print(np.unique(y))

print(f"Dataset size: {X.shape[0]} pixels")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

# Train model with class weights
edge_weight = 22.0  # You can experiment with different edge weights
class_weights = {0: 1, 1: edge_weight}
rf_model = train_model(X_train, y_train, class_weights)

# Save the trained model
save_model(rf_model, edge_weight)
'''
# Test the trained model on the result images
for img_path in RESULT_IMAGES:
    print(f"Testing on {img_path}...")
    pred_img = test_unseen_image_2(os.path.join(IMAGE_DIR, img_path), rf_model)

    # Compute and print log loss for this result imagE

    y_pred = rf_model.predict_proba(X_test)[:, 1]
    weighted_log_loss = compute_weighted_log_loss(y_test, y_pred, edge_weight)


    print(f"Log Loss : {weighted_log_loss:.4f}")

'''

import torch
import torch.nn.functional as F
import numpy as np

# Test the trained model on the result images
total_loss = 0.0
iou_scores = []

for img_path in RESULT_IMAGES:
    print(f"Testing on {img_path}...")

    # Get prediction
    pred_img = test_unseen_image_2(os.path.join(IMAGE_DIR, img_path), rf_model)

    # Compute prediction probabilities for test set
    y_pred = rf_model.predict_proba(X_test)[:, 1]

    # Convert NumPy arrays to PyTorch tensors
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

    weights = torch.ones_like(y_test_tensor)  # Default weight = 1 for all pixels
    weights[y_test_tensor == 1] = 10  # Apply higher weight to positive class (edges)

    # Compute Binary Cross-Entropy (BCE) Loss
    criterion = torch.nn.BCELoss(weight = weights)
    loss = criterion(y_pred_tensor, y_test_tensor)
    total_loss += loss.item()

    # Compute IoU for each sample
    for i in range(len(y_test)):
        iou = compute_iou(y_test[i], y_pred[i], threshold=10)
        iou_scores.append(iou)

# Compute and print final metrics
avg_loss = total_loss / len(RESULT_IMAGES)
avg_iou = sum(iou_scores) / len(iou_scores)

print(f"Final Test BCE Loss: {avg_loss:.4f}")
print(f"Final Average IoU: {avg_iou:.4f}")
