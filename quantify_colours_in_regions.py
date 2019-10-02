import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_dilation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage import measure
from scipy.ndimage import median_filter
from skimage.color import label2rgb
import os
import skimage

# use sineads images to start off with:
sinead_test_image = 'sinead/Snapshot GO samples/Older samples/GO1/Iba1 NeuN/10X RB GO top.tif'

img = Image.open(sinead_test_image)
img.load()
image_arr = np.array(img)
# red is microglia, green is astrocytes, blue is cell bodies
# higher concentrations of microglia and astrocytes near injection site?

# now got matts so will try that.....
matt_test_image = 'MAX_Iba1.tif'

matt_test_image = 'matt/matt_iba1_smaller.png'
img = Image.open(matt_test_image).convert('L')
img.load()
image_arr = np.array(img)

# try using https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html#scipy.ndimage.iterate_structure
# to increase by one each time, and just keep track of the difference in fluorescence


# really dark bits are below 20, really light bits are 105
# first need to just identify bit to start going out from
def get_segmented_image(array, marker_lower, marker_upper, filter=True, thresh=100):
    markers = np.zeros_like(array)
    markers[array < marker_lower] = 2
    markers[array > marker_upper] = 1
    elevation_map = sobel(array)
    ws = watershed(elevation_map, markers)
    plt.imshow(ws)
    plt.show()
    labels = measure.label(ws)
    plt.imshow(labels)
    plt.show()
    if filter:
        labels = skimage.morphology.remove_small_objects(labels, thresh)
    plt.imshow(labels)
    plt.show()
    return labels

segmented = get_segmented_image(image_arr, 15, 105)

# best for test image.....
markers = np.zeros_like(image_arr)
markers[image_arr < 20] = 2
markers[image_arr > 80] = 1
elevation_map = sobel(image_arr)
ws = watershed(elevation_map, markers)
plt.imshow(ws)
plt.show()
labels = measure.label(ws)
plt.imshow(labels)
plt.show()
labels_filtered = skimage.morphology.remove_small_objects(labels, 100000)
plt.imshow(labels_filtered)
plt.show()

props = measure.regionprops(labels_filtered, image_arr)
# looking at centroids, i think its props[2]
# i think the label we want should have a more similar area to convex area than others


# just try...... much better!! and faster!
markers = np.zeros_like(image_arr)
markers[image_arr < 80] = 1
labels = measure.label(markers)
plt.imshow(labels)
plt.show()
labels_filtered = skimage.morphology.remove_small_objects(labels, 500)
plt.imshow(labels_filtered)
plt.show()
labels_final = measure.label(skimage.morphology.remove_small_holes(labels_filtered, 5000))
plt.imshow(labels_final)
plt.show()
props = measure.regionprops(labels_filtered, image_arr)
injection_site = props[1]
injection_site_coords = injection_site.coords
# now need to use the coordinates to make a boolean mask to inflate
# then just sum image outside mask each time

mask = np.zeros_like(image_arr)
for x, y in injection_site_coords:
    mask[x,y] = 1

masked = np.ma.masked_array(image_arr,mask)
# then
np.sum(masked)

mask = np.zeros_like(image_arr)
for x, y in injection_site_coords:
    mask[x,y] = 1
# eed to work out smallest ditance outwards- use bbox
iterations_needed = int(np.min(np.array([np.abs(injection_site.bbox[0]- image_arr.shape[0]), np.abs(injection_site.bbox[1]- image_arr.shape[0]), np.abs(injection_site.bbox[2]- image_arr.shape[1]), np.abs(injection_site.bbox[3]- image_arr.shape[1])]))/4)
sums = []
for i in range(1, iterations_needed):
    masked = np.ma.masked_array(image_arr,mask)
    sums.append(np.sum(masked))
    mask = binary_dilation(mask, iterations=4)
    print("i: ", i, " ", np.sum(mask), " sum outside mask: ", sums[-1])
    plt.imshow(mask)
    plt.show()


# so basic idea works.... now just need to locate starting point for all images
# then set up scripts to throw on cluster
# take 10th from stack

# idea to automate finding middle- injection site is surrounded by fluorescence
# other dark sites are not

# stack is 18, so take 9

def open_image(filename):
    img = Image.open(filename).convert('L')
    img.load()
    return np.array(img)

def get_injection_site_props(image_arr, pixel_thresh=80, thresh_object=500, thresh_hole=5000):
    markers = np.zeros_like(image_arr)
    markers[image_arr < pixel_thresh] = 1
    labels = measure.label(markers)
    labels_filtered = skimage.morphology.remove_small_objects(labels, thresh_object)
    labels_final = measure.label(skimage.morphology.remove_small_holes(labels_filtered, thresh_hole))
    plt.imshow(labels_final)
    plt.show()
    props = measure.regionprops(labels_final, image_arr)
    # exclude label for entirey of image
    entire_area = image_arr.shape[0]*image_arr.shape[1]
    filtered = [p for p in props if p.area < 0.5*entire_area]
    # select one that is closest to middle
    middle_x = image_arr.shape[0]/2
    middle_y = image_arr.shape[1]/2
    center = np.array([middle_x, middle_y])
    centroids =[np.array(p.centroid) for p in filtered]
    dists = [np.linalg.norm(c-center) for c in centroids]
    index_for_injection_site = np.where(dists == np.amin(dists))[0][0]
    return filtered[index_for_injection_site]

image_arr =open_image(matt_test_image)
injection_site = get_injection_site_props(image_arr)

class IntensityResults(object):
    def __init__(self):
        self.sums = []
        self.region_intensities = []
        self.areas = []

    def average_intensity_per_region(self):
        return [int(i)/int(a) for i,a in zip(self.region_intensities, self.areas)]


def compare_intensities(image_arr, injection_site, out_filename):
    injection_site_coords = injection_site.coords
    mask = np.zeros_like(image_arr)
    for x, y in injection_site_coords:
        mask[x,y] = 1
    plt.imshow(mask)
    plt.savefig(out_filename)
    # smallest ditance outwards
    iterations_needed = int(np.min(np.array([np.abs(injection_site.bbox[0]- image_arr.shape[0]), np.abs(injection_site.bbox[1]- image_arr.shape[0]), np.abs(injection_site.bbox[2]- image_arr.shape[1]), np.abs(injection_site.bbox[3]- image_arr.shape[1])]))/4)
    print(iterations_needed)
    intensity = np.sum(image_arr)# so first intensity will actually be intensity of everything outside mask
    res = IntensityResults()
    area = np.sum(mask)
    for i in range(iterations_needed):
        masked = np.ma.masked_array(image_arr,mask)
        # intensity in region is total intensity including region - total instensity excluding region
        res.region_intensities.append(intensity-np.sum(masked))
        intensity = np.sum(masked)
        res.sums.append(intensity)
        mask = binary_dilation(mask, iterations=4)
        #print("i: ", i, " ", np.sum(mask), " sum outside mask: ", res.sums[-1], " region intensity: ", res.region_intensities[-1])
        res.areas.append(np.sum(mask)-area)
        area = np.sum(mask)
    return res

# need to take into account area of region
intensities = compare_intensities(image_arr, injection_site)
# seems to work.... first value is lower, which is probably due to mask being slightly small
# rest show expected trend


image_arr =open_image("matt/matt_neun_smaller.png")
injection_site = get_injection_site_props(image_arr)
intensities = compare_intensities(image_arr, injection_site)

image_arr =open_image("matt/matt_gfap_smaller.png")
injection_site = get_injection_site_props(image_arr)
intensities = compare_intensities(image_arr, injection_site)
# test on fullsize image to get threshold
image_arr =open_image("matt/MAX_Iba1.tif")
injection_site = get_injection_site_props(image_arr, 70, 100000, 100000)
intensities = compare_intensities(image_arr, injection_site,"matt/MAX_Iba1_max.png")

final = intensities.average_intensity_per_region()
results = "matt/max_iba1_results.csv"

with open(results, "w+") as f:
    for line in final:
        f.write(str(line)+", ")
