from skimage import filters
from skimage import io
from skimage import color
from skimage import io, measure
from skimage.morphology import skeletonize

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import probabilistic_hough_line
from skimage.draw import line as draw_line
from sklearn.cluster import KMeans
import cv2


threshold=0.00
threshold_last_detection=0.1

x_left=330
y_top=143
y_max_second_top=94
y_max_second_bottom=197

image_name="Blunt_body_reentry_shapes1.png"
image_path="shockwaves_images\\"+image_name
result_name= "calibrated_training_images\\"+image_name+"_shockwave_position_2.png"


image=io.imread(image_path)
original_shape=image.shape
#make the image grayscale to have only one channel
#disregard the transparency channel
image = image[:,:,:3]
image = color.rgb2gray(image)

#apply the sobel filter
sobel_image=filters.sobel(image)

#show the image
# io.imshow(sobel_image)
# io.show()

#I know have the edges of the image, I want to localise the shock wave by passing it through a threshold filter and then manually selecting the region of interest
sobel_image[sobel_image<threshold]=0
io.imshow(sobel_image)
io.show()

# Binarize the image (ensuring only strong edges are considered)
binary_edges = sobel_image > 0.1  # Assuming edges are already thresholded

# Label connected components
labeled_edges, num_labels = measure.label(binary_edges, return_num=True, connectivity=1)

shockwave_sample=[111,93]

#find which label is the shockwave and display it

shockwave_label=labeled_edges[shockwave_sample[0],shockwave_sample[1]]
possible_shockwave_mask=labeled_edges==shockwave_label

plt.imshow(possible_shockwave_mask)
plt.show()


#thin the lines 
thin_edges = skeletonize(possible_shockwave_mask)
# Detect lines using Hough Transform
lines = probabilistic_hough_line(thin_edges, threshold=10, line_length=30, line_gap=5)

# Create a new empty mask for the shockwave
shockwave_only_mask = np.zeros_like(possible_shockwave_mask, dtype=np.uint8)

# Define a range for the shockwave angles (adjust as needed)
min_angle, max_angle = 20, 80  # Assuming the shockwave is diagonal

# Draw only diagonal lines
for line in lines:
    (x0, y0), (x1, y1) = line
    angle = np.abs(np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi)  # Convert to degrees

    if min_angle <= angle <= max_angle or (180 - max_angle) <= angle <= (180 - min_angle):
        # plt.plot((x0, x1), (y0, y1), 'w', linewidth=2)  # For visualization
        shockwave_only_mask[y0, x0] = 1
        shockwave_only_mask[y1, x1] = 1

# Display the extracted shockwave mask
plt.figure()
plt.imshow(shockwave_only_mask, cmap='gray')
plt.title("Final Isolated Shockwave Mask")
plt.show()

# Threshold to detect bright pixels (shockwave points)
shock_wave_only_mask_binary=shockwave_only_mask>threshold_last_detection


# Get the coordinates of shockwave pixels
y_coords, x_coords = np.where(shock_wave_only_mask_binary)

#manually divide each shock wave:

topleft_area= (x_coords<x_left) & (y_coords<=y_top)
bottomleft_area= (x_coords<x_left) & (y_coords>=y_top)
topright_area= (x_coords>x_left) & (y_coords<y_top)
bottomright_area= (x_coords>x_left) & (y_coords>y_top)

inbetween_topright_area= (x_coords>x_left) & (y_coords<y_max_second_top)
inbetween_bottomright_area= (x_coords>x_left) & (y_coords>y_max_second_bottom)

#first shock in toplet and inbetween top right
first_shock_area= (topleft_area | inbetween_topright_area)
#second shock in bottomleft and inbetween bottom right
second_shock_area= (bottomleft_area | inbetween_bottomright_area)
#third shock in topright but NOT inbetween top right
third_shock_area= (topright_area & ~inbetween_topright_area)
#fourth shock in bottomright but NOT inbetween bottom right
fourth_shock_area= (bottomright_area & ~inbetween_bottomright_area)

first_shock_points=np.column_stack((x_coords[first_shock_area],y_coords[first_shock_area]))
second_shock_points=np.column_stack((x_coords[second_shock_area],y_coords[second_shock_area]))
third_shock_points=np.column_stack((x_coords[third_shock_area],y_coords[third_shock_area]))
fourth_shock_points=np.column_stack((x_coords[fourth_shock_area],y_coords[fourth_shock_area]))

shocks=[first_shock_points,second_shock_points,third_shock_points,fourth_shock_points]

# Display the extracted shockwave mask
plt.figure()
plt.scatter(first_shock_points[:,0],first_shock_points[:,1])
plt.scatter(second_shock_points[:,0],second_shock_points[:,1])
plt.scatter(third_shock_points[:,0],third_shock_points[:,1])
plt.scatter(fourth_shock_points[:,0],fourth_shock_points[:,1])
plt.title("Final Isolated Shockwave Mask")
plt.show()



# Create an empty mask for the final result
shockwave_mask = np.zeros_like(image)

# Process each cluster separately
for shock in shocks:
    for points in [shock]:
        # Extract points from this cluster
        x_sorted = np.sort(points[:, 0])  # Sort by x
        y_sorted = points[np.argsort(points[:, 0]), 1]  # Sort y accordingly

        # Interpolate between consecutive points
        m,q=np.polyfit(x_sorted,y_sorted,1)
        y_interpolated = m * x_sorted + q

        # Round the interpolated values to the nearest integer
        y_interpolated = np.round(y_interpolated).astype(int)

        # Draw the interpolated line
        for x, y in zip(x_sorted, y_interpolated):
            rr, cc = draw_line(y, x, y, x)
            shockwave_mask[rr, cc] = 1

        for i in range(len(x_sorted) - 1):
            rr, cc = draw_line(y_interpolated[i], x_sorted[i], y_interpolated[i+1], x_sorted[i+1])
            shockwave_mask[rr, cc] = 1  # Mark pixels as part of the shockwave

    


# Display the final result
plt.imshow(shockwave_mask, cmap="gray")
plt.title("Continuous Shockwave Lines")
plt.show()

#save shockwave mask as new image with same size as original image
shockwave_mask=shockwave_mask*255
shockwave_mask=shockwave_mask.astype(np.uint8)

# If needed, resize the shockwave mask to match the original image
shockwave_mask_resized = cv2.resize(shockwave_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)


#show the image
io.imshow(shockwave_mask)
io.show()


io.imsave(result_name,shockwave_mask)
