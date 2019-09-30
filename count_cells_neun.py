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


contours_dark_background = measure.find_contours(image_arr, 160)
contours_light_background = measure.find_contours(image_arr, 230)
contours = []
# actually probably don't need this, can do in oter loop
for c in contours_dark_background + contours_light_background:
    if len(c) > 15 and len(c) < 100:
        contours.append(c)

fig, ax = plt.subplots()
ax.imshow(image_arr, cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# contours is a list of pairs of coordinates
# think we need to use two thresholds- one for cells close to middle (220)
# and one for cells further out
# to remove small contours, just remove short list.
# two sets should give overlapping contours

# for contours against dark background:
# need to exclude all contours within biggest contour
# actually ditto for contours with lighter background

def biggest_contour_metrics(list_of_contours):
    max_contour = max(list_of_contours, key=len)
    xs = [i[0] for i in max_contour]
    ys = [i[1] for i in max_contour]
    max_x = np.max(xs)
    min_x = np.min(xs)
    max_y = np.max(ys)
    min_y = np.min(ys)
    return max_x, min_x, max_y, min_y

# then determine if a contour is inside

# todo: do thresholding inside list comprehension then np.any
# doesn't work cos it takes everything inside bounding box
def is_inside_contour(contour, max_x, min_x, max_y, min_y):
    xs = [i[0] for i in contour if i[0] >= min_x and i[0] <= max_x]
    ys = [i[1] for i in contour if i[1] >= min_y and i[1] <= max_y]
    for x, y in  zip(xs, ys):
        if (x >= min_x and x <= max_x) and (y >= min_y and y <= max_y):
            return True
    return False

contours_dark_background = measure.find_contours(image_arr, 160)
contours_light_background = measure.find_contours(image_arr, 230)
dark_max_x, dark_min_x, dark_max_y, dark_min_y = biggest_contour_metrics(contours_dark_background)
contours_dark_filtered = [c for c in contours_dark_background if not (is_inside_contour(c, dark_max_x, dark_min_x, dark_max_y, dark_min_y))]

light_max_x, light_min_x, light_max_y, light_min_y = biggest_contour_metrics(contours_light_background)
contours_light_filtered = [c for c in contours_dark_background if not (is_inside_contour(c, light_max_x, light_min_x, light_max_y, light_min_y))]
fig, ax = plt.subplots()
ax.imshow(image_arr, cmap=plt.cm.gray)

for n, contour in enumerate(contours_light_background):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

contours_dark_background = measure.find_contours(image_arr, 180)
dark_path = Path([(i[0], i[1]) for i in contours_dark_background[0]])
# then can do contains path....
largest_path_dark = Path([(i[0],i[1]) for i in max(contours_dark_background, key=len)])
contours = []
for c in contours_dark_background:
    if len(c) > 10 and len(c) < 80:
        path = Path([(i[0], i[1]) for i in c])
        if not largest_path_dark.contains_path(path):
            contours.append(c)

contours_light_background = measure.find_contours(image_arr, 230)
light_path = Path([(i[0], i[1]) for i in contours_light_background[0]])
# then can do contains path....
largest_path_light = Path([(i[0],i[1]) for i in max(contours_light_background, key=len)])
for c in contours_light_background:
    if len(c) > 10 and len(c) < 80:
        path = Path([(i[0], i[1]) for i in c])
        if not largest_path_light.contains_path(path):
            contours.append(c)


fig, ax = plt.subplots()
ax.imshow(image_arr, cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
# misses some obvious ones but hopefully should work
