from skimage import filters
from skimage import io

image_path="shockwaves_images\Blunt_body_reentry_shapes1.png"
image=io.imread(image_path)
#make the image grayscale to have only one channel
image=image[:,:,0] #

#apply the sobel filter
edges=filters.sobel(image)

#show the image
io.imshow(edges)
io.show()