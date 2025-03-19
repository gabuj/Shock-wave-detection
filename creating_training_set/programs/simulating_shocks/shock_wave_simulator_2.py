import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from image_shapes import image_shapes
from scipy.special import erf
import random
from PIL import Image

# Image dimensions
width, height = random.choice(image_shapes)

# Define physical parameters
rho1 = 1.0  # Upstream density
M1 = 10.03    # Upstream Mach number
gamma = 1.4 # Specific heat ratio
# rho_object=[5-10?]
# Parameters for shock front modeling
shock_peak_factor = 1.1  # How much higher the peak is compared to rho2
alpha=0

noise_level = rho1/30  # Adjust as needed
Dt=width*2 #diffusion coeff and time to take int account in difffusion
half_thickness=int(width/200) if width >200 else 1
sigma = half_thickness  # Standard deviation for peak width
centre_image=[height//2,width // 2]


#object parameters
# Random object size (ensure it fits in the image)
obj_width = random.randint(width // 20, width // 5)  # 5% to 20% of width
obj_heightop = random.randint(height // 10, height // 3)  # 10% to 33% of height
obj_heightbottom= random.randint(height // 10, height // 3)# 10% to 33% of height
#circle
r=20

def add_triangle_touching_shock(density, shock_position, rho_max,obj_width,obj_heighttop, obj_heightbottom):
    """
    Adds a random object with max density touching the shock front.

    """
    height, width = density.shape
    y_centre,x_centre=shock_position


    # Ensure object doesn't exceed boundaries
    x_end = min(width, x_centre + obj_width)
    y_top = min(height, y_centre + obj_heighttop)
    y_bottom= max(0, y_centre - obj_heightbottom)

# Triangle vertices
    A = (x_centre,y_centre)  # Tip of the triangle
    B = (x_end,y_top)  # Bottom-left
    C = (x_end,y_bottom)  # Bottom-right

# Function to check if a point is inside the triangle using Barycentric coordinates
    def is_inside_triangle(x, y, A, B, C):
        def area(x1, y1, x2, y2, x3, y3):
            return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

        total_area = area(*A, *B, *C)
        area1 = area(x, y, *B, *C)
        area2 = area(x, y, *A, *C)
        area3 = area(x, y, *A, *B)

        return abs(total_area - (area1 + area2 + area3)) < 1e-6  # Floating point tolerance

    # Iterate over the y-values from the minimum to the maximum y-coordinate of the triangle
    for x in range(min(A[0], B[0], C[0]), max(A[0], B[0], C[0]) + 1):
        if 0 <= x < width:  # Ensure within bounds
            # Iterate over the x-values from the minimum to the maximum x-coordinate of the triangle
            for y in range(min(A[1], B[1], C[1]), max(A[1], B[1], C[1]) + 1):
                if 0 <= y < height:  # Ensure within bounds
                    # Check if the point (x, y) is inside the triangle
                    if is_inside_triangle(x, y, A, B, C):
                        density[y, x] = rho_max  # Set max density for the triangle region

    return density

def add_circle_to_density(density, center, r, rho_max, alpha):
    """
    Adds a circle of radius `r` to the density field, filling it with max density.
    
    Parameters:
    - density: 2D numpy array representing the density field.
    - center: Tuple (y, x) of the center of the circle (row, column).
    - r: Radius of the circle.
    - rho_max: Maximum density to set inside the circle.
    """
    height, width = density.shape
    yc, x_shock = center  # position of shock

    d=r/np.cos(alpha)
    xc=x_shock+d


    # Iterate over all points in the density field
    for y in range(height):
        for x in range(width):
            # Calculate the squared distance from the center of the circle to the point (x, y)
            distance_squared = (y - yc)**2 + (x - xc)**2
            
            # If the distance is less than or equal to the radius squared, it's inside the circle
            if distance_squared <= r**2:
                density[y, x] = rho_max  # Set max density for the circle region

    return density



# Calculate downstream density using Rankine-Hugoniot relation
rho2 = ((gamma + 1) * M1**2) / ((gamma - 1) * M1**2 + 2) * rho1
# Create base density field
density = np.ones((height, width)) * rho1
shock_position = width // 2

rho_object=rho2*2  


# Create shock front peak
shock_peak = (shock_peak_factor * rho2)
density [:, shock_position-half_thickness:shock_position+half_thickness] = shock_peak


# Apply Gaussian smoothing to simulate shock thickness
density = gaussian_filter(density, sigma=(sigma,0), mode='nearest') # ga


# Model diffusion after the shock
# Create the diffusion profile using an error function (erf)
x = np.arange(width-shock_position-half_thickness)
diffusion_profile = 0.5 * (1 + erf((x) / np.sqrt(4*Dt)))
for i in range(height):
    # Base density: upstream density + density jump across the shock with diffusion
    density[i, shock_position+half_thickness:] += rho2 - rho1 - (rho2 - rho1) * diffusion_profile
    # Add the shock peak




# density=add_triangle_touching_shock(density, centre_image, rho_object,obj_width,obj_heightop,obj_heightbottom)
density=add_circle_to_density(density, centre_image, r, rho_object,alpha)


# Apply slight Gaussian smoothing for realism
sigma = 1.2  # Standard deviation for 
density = gaussian_filter(density, sigma=sigma, mode='nearest')

plt.plot(np.arange(width), density[2,:])
plt.show()

#the denser the darker:
density=density-density.max()
density=np.abs(density)


# Add some noise for realism (optional)
noise = np.random.normal(0, noise_level, size=density.shape)
density += noise * density  # Make noise proportional to density



# Plot the density field
plt.figure()
plt.imshow(density, cmap='grey', origin='lower')
cbar = plt.colorbar(label='Density')
plt.title('Simulated Shock Wave Density Field')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Assume `density` is your final NumPy array
normalized_density = (density - density.min()) / (density.max() - density.min())  # Normalize to [0, 1]
image_array = (normalized_density * 255).astype(np.uint8)  # Scale to [0, 255]

# Create and save the image
image = Image.fromarray(image_array)
image.save("output.jpg")
