import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage import filters
from skimage import color

threshold=0.024


# Load the image
image_name = "2005125111210_846"
image_path = "creating_training_set/shockwaves_images/" + image_name + ".jpg"
result_name = "temporary_traces/programs/" + image_name + "_shockwave_position_4.png"

image = cv2.imread(image_path)
height, width = image.shape[:2]


image = image[:,:,:3]
image = color.rgb2gray(image)

#apply the sobel filter
sobel_image=filters.sobel(image)

#I know have the edges of the image, I want to localise the shock wave by passing it through a threshold filter and then manually selecting the region of interest
sobel_image[sobel_image<threshold]=0

#reconvert sobel to rdg by putting equal values in all channels
sobel_image = np.stack([sobel_image] * 3, axis=-1)



# Create a blank mask for drawing
mask = np.zeros((height, width), dtype=np.uint8)
drawing = False
current_stroke = []  # Store points for the current stroke
all_strokes = []  # Store all drawn strokes (each stroke will be a list of points)

# Number of points to consider for curve fitting
fit_window = 15  # Allow more points for smoother fitting

# Mouse callback function
def draw_line(event, x, y, flags, param):
    global drawing, current_stroke, all_strokes

    if event == cv2.EVENT_LBUTTONDOWN:  # Start a new stroke
        drawing = True
        current_stroke = [(x, y)]  # Initialize stroke

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Avoid duplicate points
        if len(current_stroke) == 0 or (x, y) != current_stroke[-1]:
            current_stroke.append((x, y))
            
            # Keep only the last `fit_window` points for curve fitting
            if len(current_stroke) > fit_window:
                current_stroke = current_stroke[-fit_window:]

            # Ensure at least 4 points before fitting a B-spline
            if len(current_stroke) >= 4:
                points = np.array(current_stroke)
                
                # Set smoothing factor to higher value (more smoothing)
                s = 20  # Higher 's' to make the curve smoother, less tight to points
                
                # Fit spline using `splprep`. Higher 's' values allow more smoothing and curviness
                tck, u = splprep([points[:, 0], points[:, 1]], s=s, per=0)  # Smoothing spline
                
                # Sample the spline with more points for a smoother curve
                smooth_points = np.array(splev(np.linspace(0, 1, len(points) * 10), tck)).T.astype(int)  # More interpolation
                
                # Add the smoothed points to the mask without clearing the previous strokes
                for x, y in smooth_points:
                    # Only draw the points that are within the image bounds
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(mask, (x, y), radius=1, color=255, thickness=-2)

    elif event == cv2.EVENT_LBUTTONUP:  # Finish stroke
        drawing = False
        # Store the stroke for potential undo
        if len(current_stroke) > 3:  # Only store if enough points exist
            all_strokes.append(current_stroke)
            current_stroke = []  # Reset for the next stroke

# Function to undo the last trace (Ctrl+Z)
def undo_last_trace():
    global mask, all_strokes
    if len(all_strokes) > 0:
        # Remove the last stroke from the list
        last_stroke = all_strokes.pop()
        
        # Reset the mask
        mask = np.zeros_like(mask)  # Clear the mask
        
        # Re-draw all strokes except the last one
        for stroke in all_strokes:
            points = np.array(stroke)
            s = 20  # Keep smoothing factor for consistency
            tck, u = splprep([points[:, 0], points[:, 1]], s=s, per=0)
            smooth_points = np.array(splev(np.linspace(0, 1, len(points) * 10), tck)).T.astype(int)

            for x, y in smooth_points:
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(mask, (x, y), radius=1, color=255, thickness=-2)

# Set up OpenCV window
cv2.namedWindow("Trace the Shockwave")
cv2.setMouseCallback("Trace the Shockwave", draw_line)

while True:
    overlay = sobel_image.copy()
    overlay[mask > 0] = [0, 255, 0]  # Highlight traced parts in green
    cv2.imshow("Trace the Shockwave", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to save
        cv2.imwrite(result_name, mask)
        print(f"Saved smoothed mask as {result_name}")
        break
    elif key == ord('q'):  # Press 'q' to exit
        break
    elif key == 26:  # Press Ctrl+Z to undo (key code 26)
        undo_last_trace()

cv2.destroyAllWindows()
# Display the final result
import matplotlib.pyplot as plt
plt.imshow(mask, cmap="gray")
plt.title("Continuous Shockwave Lines")
plt.show()