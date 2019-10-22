from scipy.ndimage.morphology import binary_dilation
import matplotlib
from scipy.ndimage.morphology import binary_dilation
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skimage
from skimage import measure
from skimage.color import label2rgb
import os
from skimage import morphology
from skimage import exposure

# does make a difference....
def count_cells_simple(props):
    euler_numbers = [p.euler_number for p in props[1:]]
    # euler number of 1= no holes, 
    total_cells = 0
    for e in euler_numbers:
        if e == 0 or e == 1:
            total_cells += 1
        else:
            # euler number is 1 - number of holes, so number of cells is -euler_number + 1
            total_cells += abs(e) + 1
    return total_cells

    
def get_labelled_array(image_arr, gamma, binary_threshold):
    gamma_corrected = exposure.adjust_gamma(image_arr, gamma)
    binary = np.zeros_like(gamma_corrected)
    binary = gamma_corrected > binary_threshold
    labels = measure.label(binary)
    return labels

def overlaps_region(prop, labels):
    slice_in_labels = labels[prop.slice[0], prop.slice[1]]
    if np.any(slice_in_labels != 0): # nonzero value corresponds to label!!
        return set(slice_in_labels[np.where(slice_in_labels != 0)])
    return set() 


def count_cells_neun(filename):
    img = Image.open(filename).convert("L")
    img.load()
    image_arr = np.array(img)
    labels_low_contrast = get_labelled_array(image_arr, 8, 50)
    plt.imshow(labels_low_contrast)
    plt.show()
    labels_high_contrast = get_labelled_array(image_arr, 100, 200)
    props_low_contrast = measure.regionprops(labels_low_contrast, image_arr) 
    props_high_contrast = measure.regionprops(labels_high_contrast, image_arr)
    print(len(props_low_contrast))
    props_low_contrast_filtered = [p for p in props_low_contrast if len(p.coords)<=25 and len(p.coords) >1]
    props_high_contrast_filtered = [p for p in props_high_contrast if len(p.coords)<100 and len(p.coords) >1]
    print("low contrast filtered: ", len(props_low_contrast_filtered))
    print("high contrast filtered: ", len(props_high_contrast_filtered))
    show_cells_counted = np.zeros_like(image_arr)
    for prop in props_low_contrast_filtered:
        for coord in prop.coords:
            show_cells_counted[coord[0], coord[1]] = 1
    for prop in props_high_contrast_filtered:
        for coord in prop.coords:
            show_cells_counted[coord[0], coord[1]] = 2
    plt.imshow(show_cells_counted)
    plt.savefig("cells_counted.png")
    number_cells = count_cells_simple(props_low_contrast_filtered)
    print("%d cells found using naive method" % number_cells)
    print("found %d cells found in total %s"% (len(props_high_contrast_filtered)-1+number_cells, filename))
    # for cell counting, will need props of all cells counted
    return props_low_contrast_filtered + props_high_contrast_filtered


# create bitmap of centroids - then will be the same as intensity finding

def create_centroid_bitmap(x_dim, y_dim, props):
    bitmap = np.zeros((x_dim, y_dim))
    number_added = 0
    for p in props:
        centroid = p.centroid
        x_coord = int(centroid[0])
        y_coord = int(centroid[1])
        bitmap[x_coord, y_coord] = 1
        number_added += 1
    print("added %d points to bitmap" % number_added)
    return bitmap

class IntensityResults(object):
    def __init__(self):
        self.sums = []
        self.region_intensities = []
        self.areas = []

    def average_intensity_per_region(self):
        return [int(i)/int(a) for i,a in zip(self.region_intensities, self.areas)]


# base counting specific distances from hole on previous intensity counting one 
def compare_intensities(image_arr, injection_site, iteration_length, out_filename):
    injection_site_coords = injection_site.coords
    mask = np.zeros_like(image_arr)
    for x, y in injection_site_coords:
        mask[x,y] = 1
    plt.imshow(mask)
    plt.savefig(out_filename)
    pixels_per_iteration = iteration_length*4 # assume that its 4 pixels per micro (?) meter
    # smallest distance outwards
    iterations_needed = int(np.min(np.array([np.abs(injection_site.bbox[0]- image_arr.shape[0]), np.abs(injection_site.bbox[1]- image_arr.shape[0]), np.abs(injection_site.bbox[2]- image_arr.shape[1]), np.abs(injection_site.bbox[3]- image_arr.shape[1])]))/pixels_per_iteration)
    intensity = np.sum(image_arr)# so first intensity will actually be intensity of everything outside mask
    res = IntensityResults()
    area = np.sum(mask)
    for i in range(iterations_needed):
        masked = np.ma.masked_array(image_arr,mask)
        # intensity in region is total intensity including region - total instensity excluding region
        res.region_intensities.append(intensity-np.sum(masked))
        intensity = np.sum(masked)
        res.sums.append(intensity)
        mask = binary_dilation(mask, iterations=pixels_per_iteration)
        plt.imshow(mask)
        plt.show()
        res.areas.append(np.sum(mask)-area)
        area = np.sum(mask)
    return res

cell_props = count_cells_neun("matt/matt_neun_smaller.png") 
image_arr =open_image("matt/matt_neun_smaller.png")
injection_site = get_injection_site_props(image_arr, 70, 100, 200)
centroid_bitmap = create_centroid_bitmap(image_arr.shape[0], image_arr.shape[1], cell_props)
region_comparisons = compare_intensities(centroid_bitmap, injection_site, 5, "test.png")

final = region_comparisons.average_intensity_per_region()

# think it just looks like some are missing, when you zoom in, its fine
im = np.zeros_like(image_arr)
im2 = np.zeros_like(image_arr)
for p in cell_props:
    for x, y in p.coords:
        im[x,y] = 1
    im2[int(p.centroid[0]), int(p.centroid[1])] = 1

