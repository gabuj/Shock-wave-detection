import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.externals import joblib  # To save the model


# Directories
IMAGE_DIR = "creating_training_set/schockwaves_images_used"
LABEL_DIR = "creating_training_set/calibrated_training_images"
RESULT_IMAGES = ["f4_p3_cam_plane_drop_new_2-22-19.jpg",
                  "cak_colormap_0.jpg", "2005125111210_846.jpg",
                  "Screenshot from 2025-03-03 10-54-31.png",
                  "weird_fromarticle3.png", "Screenshot from 2025-03-03 10-53-39.png",
                  "airbos_f7_p5.jpg", "Screenshot from 2025-03-03 10-53-45.png"]

# Image settings
IMG_WIDTH, IMG_HEIGHT = 400, 400

# Create an output folder for the predictions if it doesn't exist
OUTPUT_FOLDER = "scikit_learning/output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_images(image_dir, label_dir, result_images, resize_shape=(400, 400)):
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
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weights, criterion='entropy')
    rf.fit(X_train, y_train)
    return rf

def save_model(model, edge_weight, filename="random_forest_model"):
    """Saves the trained model to a file with the edge weight in the filename."""
    model_filename = f"{filename}_edge_weight_{edge_weight}.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

def apply_sobel_filter(img, resize_shape=(400, 400)):
    """Apply Sobel filter to the unseen image (edge detection)."""
    img = cv2.resize(img, resize_shape)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    return sobel_edges

def test_unseen_image(image_path, model, resize_shape=(400, 400)):
    """Predicts pixel-wise classification for an unseen image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # Apply Sobel filter and flatten image for prediction
    sobel_img = apply_sobel_filter(img, resize_shape)
    img_flat = sobel_img.flatten().reshape(-1, 1)

    # Predict per pixel
    pred = model.predict(img_flat)  # Predict per pixel
    pred_img = pred.reshape(resize_shape) * 255  # Convert back to image format

    # Save output
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    pred_img = pred_img.astype(np.uint8)
    cv2.imwrite(output_path, pred_img)
    print(f"Prediction saved to {output_path}")

    return pred_img

def compute_weighted_log_loss(y_true, y_prob, edge_weight=10.0):
    """Computes weighted log loss (cross-entropy) using scikit-learn."""
    weights = np.where(y_true == 1, edge_weight, 1.0)
    loss = log_loss(y_true, y_prob, sample_weight=weights)
    return loss

# Load dataset and exclude result images
X, y = load_images(IMAGE_DIR, LABEL_DIR, RESULT_IMAGES, resize_shape=(IMG_WIDTH, IMG_HEIGHT))
print(np.unique(y))

print(f"Dataset size: {X.shape[0]} pixels")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

# Train model with class weights
edge_weight = 10.0  # You can experiment with different edge weights
class_weights = {0: 1, 1: edge_weight}
rf_model = train_model(X_train, y_train, class_weights)

# Save the trained model
save_model(rf_model, edge_weight)

# Test the trained model on the result images
for img_path in RESULT_IMAGES:
    print(f"Testing on {img_path}...")
    pred_img = test_unseen_image(os.path.join(IMAGE_DIR, img_path), rf_model)

    # Compute and print log loss for this result image
    y_true = cv2.imread(os.path.join(LABEL_DIR, os.path.splitext(img_path)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
    y_true = (y_true > 127).astype(int).flatten()
    y_prob = rf_model.predict_proba(pred_img.flatten().reshape(1, -1))[:, 1]  # Predict probabilities

    log_loss_val = compute_weighted_log_loss(y_true, y_prob, edge_weight)
    print(f"Log Loss for {img_path}: {log_loss_val:.4f}")
