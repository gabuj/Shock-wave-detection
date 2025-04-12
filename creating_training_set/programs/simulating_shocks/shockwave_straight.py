import numpy as np
import matplotlib.pyplot as plt
from image_shapes import image_shapes
from scipy.ndimage import gaussian_filter
from scipy.special import erf
import random
from PIL import Image
import cv2

#number of images
num_images= 30




rho1 = 1.0  # Upstream density
gamma = 1.4 # Specific heat ratio

#possible choices:
possible_mach_numbers=[1.01,2,3,4,5,6,7,8,9,10]
Dt_possible_values=[0.05,0.06,0.07,0.5,1,1.5,2,2.5,3,3.5,4]
possible_rho_object=[1,1.2,1.5,1.8,2,2.2,2.5]
#image destinations
# image_baseline_path="creating_training_set/simulation_images/simulated_flat_schock_"
# target_baseline_path="creating_training_set/target_simulation_images/simulated_flat_schock_"

image_baseline_path = "creating_cnn/light_inputs/simulated_schock_"
target_baseline_path = "creating_cnn/light_targets/simulated_schock_"

#define functions
def add_triangle_touching_shock(n,d_n,density, shock_position, rho_max, x_end, y_top, y_bottom):
    height, width = density.shape
    y_centre, x_centre = shock_position

    # Ensure object doesn't exceed boundaries
    # Function to check if a point is inside the triangle using Barycentric coordinates
    shock_wave_starting_points=[]
    def is_inside_triangle(x, y, A, B, C):
        def area(x1, y1, x2, y2, x3, y3):
            return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

        total_area = area(*A, *B, *C)
        area1 = area(x, y, *B, *C)
        area2 = area(x, y, *A, *C)
        area3 = area(x, y, *A, *B)

        return abs(total_area - (area1 + area2 + area3)) < 1e-6  # Floating point tolerance
    # Triangle vertices
    loop_n=0
    for loop_n in range(n):
        x_centre+=int(loop_n*d_n)
        x_end+=int(loop_n*d_n)
        if x_end> int(width*0.75):
            continue
        A = (x_centre, y_centre)  # Tip of the triangle
        B = (x_end, y_top)  # Bottom-left
        C = (x_end, y_bottom)  # Bottom-right

        shock_wave_starting_points.append(B)
        shock_wave_starting_points.append(C)
        
        # Iterate over the y-values from the minimum to the maximum y-coordinate of the triangle
        for x in range(min(A[0], B[0], C[0]), max(A[0], B[0], C[0]) + 1):
            if 0 <= x < width:  # Ensure within bounds
                # Iterate over the x-values from the minimum to the maximum x-coordinate of the triangle
                for y in range(min(A[1], B[1], C[1]), max(A[1], B[1], C[1]) + 1):
                    if 0 <= y < height:  # Ensure within bounds
                        # Check if the point (x, y) is inside the triangle
                        if is_inside_triangle(x, y, A, B, C):
                            density[y, x] = rho_max  # Set max density for the triangle region

    return density,shock_wave_starting_points

def propagate_shock(shock_x,shock_y,succ_shockwave_pos, density):
    i=0
    for x in range(width):
        if x>=shock_x:
            i+=1
            for y in range(height):
                if y<=succ_shockwave_pos[i]:
                    d=x-shock_x
                    if -tan_alpha*(y-shock_y)<d:
        
                            # Compute the normal coordinate `s` (distance along the shock normal)
                            
                        s_top = (x - shock_x) * np.sin(alpha) - (y - shock_y) * np.cos(alpha)
                
            # # Compute diffusion profile in the normal direction
                        diffusion_profile = 0.5 * (1 + erf(s_top / np.sqrt(4 * Dt)))

                        # Apply diffusion: transition from rho2 to rho1 along the normal
                        density[y, x] += rho2 - rho1 - (rho2 - rho1) * diffusion_profile
                if y>succ_shockwave_pos[i]:
                    d=x-shock_x
                    if tan_alpha*(y-shock_y)<d:
                        
                        # Compute the normal coordinate `s` (distance along the shock normal)     
                        s_bottom = (x - shock_x) * np.sin(alpha) + (y - shock_y) * np.cos(alpha)
                
            # # Compute diffusion profile in the normal direction
                        diffusion_profile = 0.5 * (1 + erf(s_bottom / np.sqrt(4 * Dt)))

                        # Apply diffusion: transition from rho2 to rho1 along the normal
                        density[y, x] += rho2 - rho1 - (rho2 - rho1) * diffusion_profile
    return density


def add_shock(shock_x,shock_y,density):
    succ_shockwave_y=[]
    if shock_y > shock_centre_y:
        top=True
    else:
        top= False
    for y in range(height):
        for x in range(width):
            if x>=shock_x:
                foundy=False
                if top and y>= shock_y:
                    # Equation of the shock line: y_shock = tan(alpha) * (x - shock_centre_x) + shock_centre_y
                    y_shock = tan_alpha * (x - shock_x) + shock_y
                    succ_shockwave_y.append(y_shock)
                    foundy=True
                if not top and y< shock_y:
                    # Equation of the shock line: y_shock = tan(alpha) * (x - shock_centre_x) + shock_centre_y
                    y_shock = -tan_alpha * (x - shock_x) + shock_y
                    succ_shockwave_y.append(y_shock)
                    foundy=True
                # Distance from (x, y) to the shock line
                if foundy:
                    distance_to_shock = abs(y - y_shock)*np.cos(alpha)
                    # If the point is within `half_thickness` of the shock, shock in target
                    if distance_to_shock <= half_thickness:
                        target[y, x] = 255
                foundy=False

    density= propagate_shock(shock_x,shock_y,succ_shockwave_y,density)
    return density

def add_successive_shocks(shock_wave_starting_points,density):
    i=0
    for i in range(len(shock_wave_starting_points)):
        succ_shock_x=shock_wave_starting_points[i][0]
        succ_shock_y=shock_wave_starting_points[i][1]
        density=add_shock(succ_shock_x,succ_shock_y,density)
    return density

def add_circle_to_density(density, rho_object, n_circles, rho1):
    """
    Adds `n_circles` with random sizes and positions to the density field.
    
    Conditions:
    - Radius is a random value between 10% and 25% of the image width.
    - Circles are placed only where at least 80% of the area has density in range [rho1 - rho1/3, rho1 + rho1/3] or exactly rho_object.
    - If a circle is partially outside the image, only the inside part is drawn.

    Parameters:
    - density: 2D numpy array representing the density field.
    - rho_object: Density value assigned to circles.
    - n_circles: Number of circles to add.
    - rho1: Base density value.

    Returns:
    - Updated density field with circles added.
    """
    height, width = density.shape
    
    for _ in range(n_circles):
        # Random radius between 10% and 25% of the image width
        r = random.randint(int(width * 0.01), int(width * 0.15))
        
        # Try random positions until a valid one is found
        for _ in range(100):  # Avoid infinite loops, max 100 tries
            yc = random.randint(r, height - r)  # Ensure it fits inside
            xc = random.randint(r, width - r)
            
            # Get the circle mask
            y, x = np.ogrid[:height, :width]
            mask = (x - xc)**2 + (y - yc)**2 <= r**2
            
            # Check if at least 80% of the circle region satisfies conditions
            circle_values = density[mask]
            valid_values = ((rho1 - rho1 / 3) <= circle_values) & (circle_values <= (rho1 + rho1 / 3)) | (circle_values == rho_object)
            if np.mean(valid_values) >= 0.95:  # If 95% or more are valid
                density[mask] = rho_object
                break  # Place the circle and move to the next one

    return density


used_parameters=[]
cycle_n=0

keep_vars = set(locals().keys()).copy()

while True:
    if cycle_n > num_images:
        break
    current_parameters=[]
    print(f"currently at image {cycle_n}")
    image_path = f"{image_baseline_path}{cycle_n}.jpg"
    target_path = f"{target_baseline_path}{cycle_n}.png"

    # Image dimensions
    width, height = random.choice(image_shapes)

    current_parameters.append(width)
    current_parameters.append(height)

    half_thickness=int(width / 300) if width > 200 else 1
    sigma = half_thickness # Standard deviation for target shock

    # Define physical parameters
    M1 = random.choice(possible_mach_numbers)    # Upstream Mach number   RANDOMISE BETWEEN 1.01 AND 10
    current_parameters.append(M1)
    # Parameters for shock front modeling
    alpha=np.pi/2 #Randomise from 0.2 to np.pi/2
    tan_alpha = np.tan(alpha)  # Slope of shock     

    # Noise and diffusion parameters
    noise_level = rho1 / random.randrange(3,39,4)  # Adjust as needed     RANDOMISE FROM 3 to 40
    current_parameters.append(noise_level)
    
    end_factor=10
    start_random=width//600
    end_random=width*end_factor//600
    steps=(end_random-start_random)//10
    steps = steps if steps>0 else 1

    #last_gaussian_convolution_sigma = random.randrange(start_random,end_random,steps)  # (1,17,4)Standard deviation for Gaussian smoothing    RANDOMISE FROM 1 TO 17
    last_gaussian_convolution_sigma = random.randrange(0, 2)  # Standard deviation for Gaussian smoothing    RANDOMISE FROM 1 TO 17
    current_parameters.append(last_gaussian_convolution_sigma)
    
    Dt = width * random.choice(Dt_possible_values)  # Diffusion coefficient and time   RANDOMISE 0.05 and 4
    current_parameters.append(Dt)

    centre_image = [height // 2, width // 2]  # Center of the image
    shock_centre_x=width//2
    shock_centre_y=height//2

    # Object parameters (random size)
    n_circles=random.randint(0,4) #randomise between 0 and 4
    current_parameters.append(n_circles)

    #create triangles
    # rho_object=[rho2*1.1-rho2*1.5] --> define later
    n=0 #randomise between 1 and 6
    
    inloop=True
    while inloop == True:
        x_end  = random.randint(int(shock_centre_x+width*0.05), int(width*0.75))  # Object width (5%-100%)  RANDOMISE
        y_top  = random.randint(shock_centre_y+height // 100, shock_centre_y+height // 3)  # Object height top  RANDOMISE
        y_bottom = random.randint(shock_centre_y-height // 3, shock_centre_y-height // 100)  # Object height bottom RANDOMISE
        cat1=abs(x_end-shock_centre_x)

        if abs(y_top-shock_centre_y)/cat1>tan_alpha:
            continue
        if abs(y_bottom-shock_centre_y)/cat1<=tan_alpha:
            inloop=False

    current_parameters.append(x_end)
    current_parameters.append(y_top)
    current_parameters.append(y_bottom)

    d_n=cat1*random.uniform(0.3,0.5) #pick random from 0.3 to 0.5
    d_n=round(d_n,1)
    current_parameters.append(d_n)

    # Calculate downstream density using Rankine-Hugoniot relation
    conv_factor=((gamma + 1) * M1**2) / ((gamma - 1) * M1**2 + 2)
    rho2 = conv_factor * rho1

    #rho object
    rho_object=rho2*random.choice(possible_rho_object) #RANDOMISE from rho2*(from 1 to 3)   
    current_parameters.append(rho_object)

    #check that not used these parameters before
    if current_parameters in used_parameters:
        print("already chosen these parameters")
        continue

    used_parameters.append(current_parameters)
    cycle_n +=1
    
    # Create base density field
    density=0 #set to 0
    density = np.ones((height, width)) * rho1
    target= np.ones((height, width)) * 0


    top_shockwave_y=[]
    bottom_shockwave_y=[]
    
    y_shock =0 #set to 0
    for y in range(height):
        for x in range(width):
            if x>=shock_centre_x:
                if y>= shock_centre_y:
                    # Equation of the shock line: y_shock = tan(alpha) * (x - shock_centre_x) + shock_centre_y
                    y_shock = tan_alpha * (x - shock_centre_x) + shock_centre_y
                    top_shockwave_y.append(y_shock)
                if y< shock_centre_y:
                    # Equation of the shock line: y_shock = tan(alpha) * (x - shock_centre_x) + shock_centre_y
                    y_shock = -tan_alpha * (x - shock_centre_x) + shock_centre_y
                    bottom_shockwave_y.append(y_shock)
                # Distance from (x, y) to the shock line
                distance_to_shock = abs(y - y_shock)*np.cos(alpha)
                
                # If the point is within `half_thickness` of the shock, assign target shock
                if distance_to_shock <= half_thickness:
                    target[y, x] = 255


    # Model diffusion after the shock
    # Create the diffusion profile using an error function (erf)
    i=0
    for x in range(width):
            if x>=shock_centre_x:
                i+=1
                for y in range(height):
                    if y<=top_shockwave_y[i]:
                        d=x-shock_centre_x
                        if -tan_alpha*(y-shock_centre_y)<d:
            
                                # Compute the normal coordinate `s` (distance along the shock normal)
                                
                            s_top = (x - shock_centre_x) * np.sin(alpha) - (y - shock_centre_y) * np.cos(alpha)
                    
                # # Compute diffusion profile in the normal direction
                            diffusion_profile = 0.5 * (1 + erf(s_top / np.sqrt(4 * Dt)))

                            # Apply diffusion: transition from rho2 to rho1 along the normal
                            density[y, x] += rho2 - rho1 - (rho2 - rho1) * diffusion_profile
                    if y>bottom_shockwave_y[i]:
                        d=x-shock_centre_x
                        if tan_alpha*(y-shock_centre_y)<d:
                            
                            # Compute the normal coordinate `s` (distance along the shock normal)     
                            s_bottom = (x - shock_centre_x) * np.sin(alpha) + (y - shock_centre_y) * np.cos(alpha)
                    
                # # Compute diffusion profile in the normal direction
                            diffusion_profile = 0.5 * (1 + erf(s_bottom / np.sqrt(4 * Dt)))

                            # Apply diffusion: transition from rho2 to rho1 along the normal
                            density[y, x] += rho2 - rho1 - (rho2 - rho1) * diffusion_profile



    # Add a triangle object touching the shock
    density,shock_wave_starting_points = add_triangle_touching_shock(n,d_n,density, centre_image, rho_object, x_end, y_top, y_bottom)

    density=add_successive_shocks(shock_wave_starting_points,density)


    density= add_circle_to_density(density, rho_object, n_circles, rho1)


    # Apply slight Gaussian smoothing for realism
    density = gaussian_filter(density, sigma=last_gaussian_convolution_sigma, mode='nearest')

    # # Plot the result
    # plt.plot(np.arange(width), density[2, :])
    # plt.show()

    # Adjust density for visualization (denser areas appear darker)
    density = density - density.max()
    density = np.abs(density)

    # Add some noise for realism (optional)
    noise = np.random.normal(0, noise_level, size=density.shape)
    density += noise * density  # Make noise proportional to density

    #add bias that will NOT be inlcuded in the random variables:
    random_bias=random.uniform(0, 0.1)
    density +=random_bias
    # # Plot the final density field
    # plt.figure()
    # plt.imshow(density, cmap='gray', origin='lower')
    # plt.title('Simulated Oblique Shock Wave Density Field')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()


    # #show target
    # plt.figure()
    # plt.imshow(target, cmap='gray', origin='lower')
    # plt.title('target Shock Wave Density Field')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()

    # Assume `density` is your final NumPy array
    normalized_density = (density - density.min()) / (density.max() - density.min())  # Normalize to [0, 1]
    image_array = (normalized_density * 255).astype(np.uint8)  # Scale to [0, 255]

    # Create and save the image
    image = Image.fromarray(image_array)
    image.save(image_path)

    target=target.astype(np.uint8)
    cv2.imwrite(target_path, target)

    #delete all local variables
    vars_to_delete = [var for var in locals().keys() if var not in keep_vars and var !='keep_vars']
    for var in vars_to_delete:
        del locals()[var]


#fake variable to run program
run_shock_waves_straight=0
