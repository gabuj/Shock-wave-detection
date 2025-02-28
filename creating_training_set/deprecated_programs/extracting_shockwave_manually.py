from skimage import filters
from skimage import io
from skimage import color

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line as draw_line
import cv2


threshold=0.01
threshold_last_detection=0.1


image_name="airbos_f7_p5"
image_path="shockwaves_images\\"+image_name + ".jpg"
result_name= "calibrated_training_images\\"+image_name+"_shockwave_position.png"


image=io.imread(image_path)
original_shape=image.shape
#make the image grayscale to have only one channel
#disregard the transparency channel
image = image[:,:,:3]
image = color.rgb2gray(image)

#apply the sobel filter
sobel_image=filters.sobel(image)


plt.figure()
plt.imshow(sobel_image,cmap="gray")
plt.title("Sobel Filtered Image")
plt.show()





shock1=[[625,1860],[728,4306]]  #y1,x1, y2,x2

shock2=[[625,1860],[1574,2]]  #y1,x1, y2,x2

shock3=[[1440,2078],[1733,4305]]  #y1,x1, y2,x2

shock4=[[1168, 2214],[1220, 2362]]  #y1,x1, y2,x2
shock24=[[1220, 2362],[1240, 2478]]  #y1,x1, y2,x2
shock25=[[1240, 2478],[1274, 2630]]  #y1,x1, y2,x2
shock26=[[1274, 2630],[1304, 2821]]  #y1,x1, y2,x2
shock27=[[1304, 2821],[1371, 3248]]  #y1,x1, y2,x2
shock7=[[1371, 3248],[1424, 3615]]  #y1,x1, y2,x2
shock8=[[1424, 3615],[1515, 4297]]  #y1,x1, y2,x2

shock9=[[1293, 2066],[1360, 2772]]  #y1,x1, y2,x2
shock10=[[1360, 2772],[1515, 4297]]  #y1,x1, y2,x2

shock11=[[1315, 1954],[1539, 1564]]  #y1,x1, y2,x2
shock12=[[1539, 1564],[2091, 628]]  #y1,x1, y2,x2

shock16=[[1272, 1751],[1348, 1680]]  #y1,x1, y2,x2
shock17=[[1348, 1680],[1394, 1629]]  #y1,x1, y2,x2
shock13=[[1394, 1629],[1438, 1568]]  #y1,x1, y2,x2
shock14=[[1438, 1568],[1516, 1463]]  #y1,x1, y2,x2
shock15=[[1516, 1463],[1593, 1346]]  #y1,x1, y2,x2
shock18=[[1593, 1346],[1706, 1183]]  #y1,x1, y2,x2
shock19=[[1706, 1183],[2397, 122]]  #y1,x1, y2,x2

shock20=[[1041, 2012],[1294, 4301]]  #y1,x1, y2,x2

shock21=[[1068, 1887],[2202, 2]]  #y1,x1, y2,x2

shock22=[[942, 1963],[1200, 4303]]  #y1,x1, y2,x2

shock23=[[959, 1894],[2099, 2]]  #y1,x1, y2,x2

shock28=[[1445, 1990],[1512, 1911]]  #y1,x1, y2,x2
shock29=[[1512, 1911],[1602, 1776]]  #y1,x1, y2,x2
shock30=[[1602, 1776],[2397, 541]]  #y1,x1, y2,x2

shocks=[shock1,shock2,shock3,shock4,shock7,shock8,shock9,shock10,shock11,shock12,shock13,shock14,shock15,shock16,shock17,shock18,shock19,shock20,shock21,shock22,shock23,shock24,shock25,shock26,shock27,shock28,shock29,shock30]
# Create an empty mask for the final result
shockwave_mask = np.zeros_like(image)

# Process each cluster separately
for shock in shocks:
    x1=shock[0][1]#1,x
    y1=shock[0][0]
    x2=shock[1][1]
    y2=shock[1][0]
    # Compute the line between the two points
    y, x = draw_line(y1, x1, y2, x2)
    shockwave_mask[y, x] = 1


    


# Display the final result
plt.imshow(shockwave_mask, cmap="gray")
plt.title("Continuous Shockwave Lines")
plt.show()

#save shockwave mask as new image with same size as original image
shockwave_mask=shockwave_mask*255
shockwave_mask=shockwave_mask.astype(np.uint8)


#show the image
io.imshow(shockwave_mask)
io.show()


io.imsave(result_name,shockwave_mask)
