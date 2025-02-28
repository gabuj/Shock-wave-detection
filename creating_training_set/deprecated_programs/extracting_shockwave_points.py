from skimage import filters
from skimage import io
from skimage import color

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line as draw_line
import cv2


threshold=0.01
threshold_last_detection=0.1


image_name="airbos_f7_p5.jpg"
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

# Function to handle the mouse click event
def onclick(event):
    # Get the x and y coordinates of the click
    x, y = int(event.xdata), int(event.ydata)
    
    # Ensure coordinates are within image bounds
    if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
        print(f"[{y}, {x}]")
    else:
        print("Click was outside of the image bounds!")

# Plot the Sobel image
plt.figure()
plt.imshow(sobel_image, cmap="gray")
plt.title("Sobel Filtered Image")

# Connect the mouse click event to the onclick function
plt.gcf().canvas.mpl_connect('button_press_event', onclick)

plt.show()