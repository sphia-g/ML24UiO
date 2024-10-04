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

'''''
# Downsample the data by selecting every 100th point
downsample_factor = 100
terrain1_downsampled = terrain1[::downsample_factor, ::downsample_factor]

# Create meshgrid
ysize, xsize = terrain1_downsampled.shape
x = np.linspace(0, xsize - 1, xsize)
y = np.linspace(0, ysize - 1, ysize)
x, y = np.meshgrid(x, y)

'''''

# Flatten x and y to create a design matrix
x_flat = x.flatten()
y_flat = y.flatten()