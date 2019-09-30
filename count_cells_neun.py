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
from skimage import morphology
from pyamg.graph import vertex_coloring
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

contours_dark_background = measure.find_contours(image_arr, 140)
dark_path = Path([(i[0], i[1]) for i in contours_dark_background[0]])
# then can do contains path....
largest_path_dark = Path([(i[0],i[1]) for i in max(contours_dark_background, key=len)])
contours = []
for c in contours_dark_background:
    if len(c) > 10 and len(c) < 80:
        path = Path([(i[0], i[1]) for i in c])
        if not largest_path_dark.contains_path(path):
            contours.append(c)

binary = np.zeros_like(image_arr)
for c in contours:
    for p in c:
        binary[int(p[0]), int(p[1])] = 1

labels = measure.label(binary)
plt.imshow(labels)
plt.show()

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

# try just making everything inside a contour black, rest white
binary = np.zeros_like(image_arr)
for c in contours:
    for p in c:
        binary[int(p[0]), int(p[1])] = 1

labels = measure.label(binary)
plt.imshow(labels)
plt.show()

# just try standard labelling technique....

markers = np.zeros_like(image_arr)
markers[image_arr > 180] = 2
markers[image_arr < 80] = 1
plt.imshow(markers)
plt.show()

labels = measure.label(markers)
plt.imshow(labels)
plt.show()
elevation_map = scharr(image_arr)
ws = watershed(elevation_map, markers)
plt.imshow(ws)
plt.show()
labels = measure.label(ws)
plt.imshow(labels)
plt.show()
labels_filtered = skimage.morphology.remove_small_objects(labels, 100000)
plt.imshow(labels_filtered)
plt.show()

# just try some different filters....
# this has potential.....
t = skimage.filters.apply_hysteresis_threshold(image_arr, 170, 190)
plt.imshow(t)
plt.show()

thresholds = skimage.filters.threshold_multiotsu(image_arr)
# Using the threshold values, we generate the three regions.
regions = np.digitize(image, bins=thresholds)

optimal_threshold = skimage.filters.threshold_li(image_arr)

plt.imshow(image_arr > optimal_threshold, cmap='gray')
plt.show()


# http://emmanuelle.github.io/a-tutorial-on-segmentation.html

from skimage import restoration
from skimage import img_as_float
im_float = img_as_float(img)

im_denoised = skimage.restoration.denoise_nl_means(im_float, h=0.05)
plt.imshow(im_denoised, cmap='gray')
ax = plt.axis('off')
plt.show()


# Try to threshold the image
plt.imshow(im_denoised, cmap='gray')
plt.contour(im_denoised, [0.5], colors='yellow')
plt.contour(im_denoised, [0.45], colors='blue')
ax = plt.axis('off')

plt.show()
from skimage import feature
edges = feature.canny(im_denoised, sigma=0.2, low_threshold=0.07, \
                      high_threshold=0.18)
plt.imshow(im_denoised, cmap='gray')
plt.contour(edges)

hat = ndimage.black_tophat(im_denoised, 7)
# Combine with denoised image
hat -= 0.3 * im_denoised
# Morphological dilation to try to remove some holes in hat image
hat = morphology.dilation(hat)
plt.imshow(hat, cmap='viridis')

labels_hat = morphology.watershed(hat, markers)
from skimage import color
color_labels = color.label2rgb(labels_hat, im_denoised)
plt.imshow(color_labels)
plt.show()

# A different markers image: laplace filter
lap = ndimage.gaussian_laplace(im_denoised, sigma=0.7)
lap = restoration.denoise_nl_means(lap, h=0.002)
plt.imshow(lap[:300, :300], cmap='viridis'); plt.colorbar()
plt.show()

def trim_close_points(points, distance=1):
    """
    Greedy method to remove some points so that
    all points are separated by a distance greater
    than ``distance``.
    
    points : array of shape (2, n_points)
        Coordinates of 2-D points
        
    distance : float
        Minimal distance between points
    """
    x, y = points
    tree = spatial.KDTree(np.array([x, y]).T)
    pairs = tree.query_pairs(distance)
    remove_indices = []
    for pair in pairs:
        if pair[0] in remove_indices:
            continue
        if pair[1] in remove_indices:
            continue
        else:
            remove_indices.append(pair[1])
    keep_indices = np.setdiff1d(np.arange(len(x)), remove_indices)
    return np.array([x[keep_indices], y[keep_indices]])

from sklearn.neighbors import kneighbors_graph
from skimage import segmentation
import pyamg 
from scipy import spatial
n_real = 400
n_markers = 1000
segmentations = []
for real in range(n_real):
    # Random markers
    x, y = np.random.random((2, n_markers))
    x *= im_denoised.shape[0]
    y *= im_denoised.shape[1]
    # Remove points too close to each other
    xk, yk = trim_close_points((x, y), 10)
    mat = kneighbors_graph(np.array([xk, yk]).T, 12)
    colors = vertex_coloring(mat)
    # Array of markers
    markers_rw = np.zeros(im_denoised.shape, dtype=np.int)
    markers_rw[xk.astype(np.int), yk.astype(np.int)] = colors + 1
    markers_rw = morphology.dilation(markers_rw, morphology.disk(3))
    # Segmentation
    labels_rw = segmentation.random_walker(im_denoised[::2, ::2], 
                                           markers_rw[::2, ::2],\
                                       beta=25000, mode='cg_mg')
    segmentations.append(labels_rw)


from skimage import measure, color
############# WHY ARE ALL THESE BLANK!!!!!!!!
segmentations = np.array(segmentations)
boundaries = np.zeros_like(im_denoised[::2, ::2])
for seg in segmentations:
    boundaries += segmentation.find_boundaries(seg, connectivity=2)


plt.imshow(boundaries, cmap='gist_heat'); plt.colorbar()

plt.show()

def hysteresis_thresholding(im, v_low, v_high):
    """
    Parameters
    ----------
    im : 2-D array
    
    v_low : float
        low threshold
        
    v_high : float
        high threshold
    """
    mask_low = im > v_low
    mask_high = im > v_high
    # Connected components of mask_low
    labels_low = measure.label(mask_low, background=0) + 1
    count = labels_low.max()
    # Check if connected components contain pixels from mask_high
    sums = ndimage.sum(mask_high, labels_low, np.arange(count + 1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums[1:] > 0
    output_mask = good_label[labels_low]
    return output_mask 



def color_segmentation(regions, n_neighbors=25):
    """
    Reduce the number of labels in a label image to make
    visualization easier.
    """
    count = regions.max()
    centers = ndimage.center_of_mass(regions + 2, regions, 
                                     index=np.arange(1, count + 1))
    centers = np.array(centers)
    mat = kneighbors_graph(np.array(centers), n_neighbors)
    colors = vertex_coloring(mat)
    colors = np.concatenate(([0], colors))
    return colors[regions]                       


def plot_colors(val_low, val_high):
    """
    Plot result of segmentation superimposed on original image,
    and plot original image as well.
    """
    seg = hysteresis_thresholding(boundaries, val_low, val_high)
    regions = measure.label(np.logical_not(seg),
                            background=0, connectivity=1)
    color_regions = color_segmentation(regions)
    colors = [plt.cm.spectral(val) for val in 
                   np.linspace(0, 1, color_regions.max() + 1)]
    image_label_overlay = color.label2rgb(color_regions, 
                                          im_denoised[::2, ::2],
                                          colors=colors)
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(image_label_overlay)
    plt.subplot(122)
    plt.imshow(im_denoised, cmap='gray')
    return regions




