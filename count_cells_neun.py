import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_dilation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from skimage.filters import sobel
#from skimage.morphology import watershed
from skimage import measure
#from scipy.ndimage import median_filter
from skimage.color import label2rgb
import os
#from skimage import feature, color
#from skimage.transform import hough_circle
#from skimage.feature import peak_local_max
#from skimage.draw import circle_perimeter


img = Image.open("matt/matt_neun_smaller.png").convert("L")
img.load()
image_arr = np.array(img)

binary = np.array([[image_arr[i,j] > 200 for i in range(image_arr.shape[0])] for j in range(image_arr.shape[1])])

# not going to be able to use binary image for identifying overlapping bits
# may be better to use full rgb version? or are they always equal in which case its just more computation

# average radius seems to be about 2.5 pixels
edges = feature.canny(image_arr, sigma=0.05, low_threshold=155, high_threshold=160)

# don't think they are circular enough for hough transform, but lets try
# if not, remove small flecks and label

hough_radii = np.array([1.5, 3])
hough_res = hough_circle(edges, hough_radii)

centers = []
accums = []
radii = []

for radius, h in zip(hough_radii, hough_res):
    # For each radius, extract two circles
    peaks = peak_local_max(h, num_peaks=2)
    centers.extend(peaks - hough_radii.max())
    accums.extend(h[peaks[:, 0], peaks[:, 1]])
    radii.extend([radius, radius])

# Draw the most prominent 5 circles
image = color.gray2rgb(image_arr)
for idx in np.argsort(accums)[::-1]:
    center_x, center_y = centers[idx]
    radius = radii[idx]
    cx, cy = circle_perimeter(int(center_y), int(center_x), int(radius))
    image[cy, cx] = (220, 20, 20)

plt.imshow(image, cmap=plt.cm.gray)
plt.show()
# ha! one of the only circles it found was not a cell

# try contour finding instead of edge detection
