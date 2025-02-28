import cv2
import numpy as np
import os

# Path to directory with multiple trace images
trace_dir = "temporary_traces/"
trace_files = os.listdir(trace_dir)  # List of all trace files (assumed to be images)

image_name = "200512510409_846"
result_name = "calibrated_training_images/" + image_name + "_shockwave_position_1.png"

threshold=127

# Assuming all images are of the same size
# Initialize an empty array to store the summed pixel values
first_image = cv2.imread(os.path.join(trace_dir, trace_files[0]), cv2.IMREAD_GRAYSCALE)
height, width = first_image.shape

# Initialize a sum array for averaging
sum_mask = np.zeros((height, width), dtype=np.float32)

trace_images = []
# Loop through all the trace images and accumulate their pixel values
for trace_file in trace_files:
    trace_path = os.path.join(trace_dir, trace_file)
    trace_image = cv2.imread(trace_path, cv2.IMREAD_GRAYSCALE)
    trace_images.append(trace_image)
    sum_mask += trace_image

combined_mask = np.median(np.array(trace_images), axis=0).astype(np.uint8)


# # Calculate the average by dividing by the number of images
num_traces = len(trace_files)
average_mask = sum_mask / num_traces

# Threshold the averaged result to create a final binary mask
thresholded_mask = np.where(combined_mask > threshold, 255, 0).astype(np.uint8)

# display the result
import matplotlib.pyplot as plt
plt.imshow(thresholded_mask, cmap='gray')
plt.show()

#save
cv2.imwrite(result_name, thresholded_mask)