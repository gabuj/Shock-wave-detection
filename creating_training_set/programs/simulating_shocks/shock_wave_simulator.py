import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from image_shapes import image_shapes
import random

# Image dimensions
width, height = random.choice(image_shapes)

# Define densities
rho1 = 1.0  # Upstream density
M1 = 2.0    # Upstream Mach number
gamma = 1.4 # Specific heat ratio
rho2 = ((gamma + 1) * M1**2) / ((gamma - 1) * M1**2 + 2) * rho1  # Downstream density

# Create density field
density = np.ones((height, width)) * rho1
shock_position = width // 2
density[:, shock_position:] = rho2

# Apply Gaussian smoothing to simulate shock thickness
sigma = 2  # Standard deviation for Gaussian kernel
density = gaussian_filter(density, sigma=sigma, mode='nearest')

# Plot the density field
plt.imshow(density, cmap='gray', origin='lower')
plt.colorbar(label='Density')
plt.title('Simulated Shock Wave Density Field')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()