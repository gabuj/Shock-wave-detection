import cv2
import numpy as np
from skimage import filters, color
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

'''#training images path: 

/Users/andracriscov/Documents/project Y3/repo/creating_training_set/schockwaves_images_used/200512510409_846.jpg
airbos_f7_p5.jpg
Blunt_body_reentry_shapes1.png
'''
# Define paths
trace_dir = "creating_training_set/shockwaves_images/"
trace_files = os.listdir(trace_dir)

threshold = 0.024
sample_size = 10000  #number of pixels to sample


features = []
labels = []

for image_file in trace_files:
    image_path = os.path.join(trace_dir, image_file)
    image = cv2.imread(image_path)

    if image is None:
        continue

    #convert to grayscale and apply Sobel filter
    image = color.rgb2gray(image)
    sobel_image = filters.sobel(image)

    # Threshold edges
    binary_edges = (sobel_image > threshold).astype(np.uint8)

    #randomly selecting pixels for training
    height, width = binary_edges.shape
    for _ in range(sample_size):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        patch = image[max(0, y - 1): min(height, y + 2), max(0, x - 1): min(width, x + 2)]
        patch = patch.flatten()

        if len(patch) == 9:  # Ensure full patch size
            features.append(patch)
            labels.append(binary_edges[y, x])  # 1 for edge, 0 for background


X = np.array(features)
y = np.array(labels)

print(f"Dataset size: {X.shape[0]} samples, each with {X.shape[1]} features.")



'''model'''

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#model performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


def detect_edges(image_path, model, threshold=0.024):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = color.rgb2gray(image)
    height, width = image.shape
    sobel_image = filters.sobel(image)

    predicted_mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            patch = image[y - 1:y + 2, x - 1:x + 2].flatten()
            if len(patch) == 9:
                pred = model.predict([patch])[0]
                predicted_mask[y, x] = 255 if pred == 1 else 0  # White for edges

    return predicted_mask


# Test the model on a new image
test_image = "/Users/andracriscov/Documents/project Y3/repo/creating_training_set/shockwaves_images/train_1.png"
predicted_edges = detect_edges(test_image, clf)

print(type(predicted_edges))  # gives type none
print(predicted_edges.shape)

# Display the result
plt.imshow(predicted_edges, cmap="gray")
plt.title("AI-Detected Shockwave Edges")
plt.show()

#Load manually traced mask
manual_mask = cv2.imread("/Users/andracriscov/Documents/project Y3/repo/creating_training_set/shockwaves_images/label_1.png", cv2.IMREAD_GRAYSCALE)
manual_mask = cv2.threshold(manual_mask, 127, 255, cv2.THRESH_BINARY)[1] // 255  # Convert to binary

# Convert AI prediction to binary
ai_mask = predicted_edges // 255

# Calculate precision and recall
precision = precision_score(manual_mask.flatten(), ai_mask.flatten())
recall = recall_score(manual_mask.flatten(), ai_mask.flatten())

#precision = precision_score(ai_mask.flatten())
#recall = recall_score(ai_mask.flatten())

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")





# Display edges for each image
plt.figure(figsize=(6, 6))
plt.imshow(binary_edges, cmap="gray")
plt.title(f"Binary Edges: {image_file}")
plt.axis("off")
plt.show()


