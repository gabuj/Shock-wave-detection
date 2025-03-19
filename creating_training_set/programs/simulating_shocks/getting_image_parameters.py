import numpy as np
import os
import cv2
from skimage import color
images_dir="creating_training_set/schockwaves_images_used/"

#get images
image_files = os.listdir(images_dir)

image_shapes=[]
for path in image_files:
    full_path= images_dir+path
    image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
    shape=[image.shape[0],image.shape[1]]
    image_shapes.append(shape) if shape not in image_shapes else print("shape already there")

print(image_shapes)