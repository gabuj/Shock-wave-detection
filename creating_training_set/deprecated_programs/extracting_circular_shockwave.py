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


threshold=0.2
threshold_last_detection=0.1

x_left=330
y_top=143
y_max_second_top=94
y_max_second_bottom=197

image_name="200512510347_846"
image_path="shockwaves_images\\"+image_name + ".jpg"
result_name= "calibrated_training_images\\"+image_name+"_shockwave_position.png"


image=io.imread(image_path)
original_shape=image.shape
#make the image grayscale to have only one channel
#disregard the transparency channel
image = image[:,:,:3]
image = color.rgb2gray(image)

#convert image to uint8
#normalise the image
image=image/image.max()
image=image*255
image=image.astype(np.uint8)


circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=300,
                           param1=90, param2=90, minRadius=50, maxRadius=200)

# Create a binary mask of the detected circle
shockwave_mask = np.zeros_like(image, dtype=np.uint8)

# Draw only the circle perimeters
if circles is not None:
    circles = np.uint16(np.around(circles))  # Convert to integer
    for circle in circles[0, :]:  # Iterate over detected circles
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(shockwave_mask, center, radius, 255, thickness=1)  # Perimeter only

# Show results
plt.figure(figsize=(10, 5))
plt.imshow(shockwave_mask, cmap='gray')
plt.title("Detected Circular Shockwave Mask")
plt.show()

#save shockwave mask as new image with same size as original image
shockwave_mask=shockwave_mask*255
shockwave_mask=shockwave_mask.astype(np.uint8)

#show the image
io.imshow(shockwave_mask)
io.show()


io.imsave(result_name,shockwave_mask)
