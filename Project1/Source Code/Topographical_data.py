import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Use data points from TIFF file
terrain1_array = np.array(terrain1)
# Get image dimensions
height, width = terrain1_array.shape[:2]
# Create meshgrid for X and Y coordinates
x, y = np.meshgrid(range(width), range(height))

# Flatten x and y to create a design matrix
x_flat = x.flatten()
y_flat = y.flatten()