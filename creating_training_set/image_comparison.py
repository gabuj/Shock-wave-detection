import matplotlib.pyplot as plt
import cv2
from skimage import color

image_name="Blunt_body_reentry_shapes1.png"
image="shockwaves_images\\"+image_name
result= "calibrated_training_images\\"+image_name+"_shockwave_position.png"


# Load images
image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
result = cv2.imread(result, cv2.IMREAD_UNCHANGED)
#make the image grayscale to have only one channel
#disregard the transparency channel
image = image[:,:,:3]
image = color.rgb2gray(image)

#normalise the image
image=image/image.max()
#normalise the result
result=result/result.max()

#change result dtype to match image
result = result.astype(image.dtype)
print(result.dtype)
# Blend images (adjust alpha for transparency)
alpha = 0.5  # 50% transparency
blended = cv2.addWeighted(image, alpha, result, 1-alpha, 0)

# Show comparison
plt.figure(figsize=(10,5))
plt.imshow(blended, cmap='gray')
plt.title("Overlayed Image")
plt.show()