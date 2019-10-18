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
    # large low contrast labels are likely to be multiple cells 
    props_low_contrast_large = [p for p in props_low_contrast if len(p.coords)>25 and len(p.coords) < 100]
    props_low_contrast_filtered = [p for p in props_low_contrast if len(p.coords)<=25 and len(p.coords) >1]
    props_high_contrast_filtered = [p for p in props_high_contrast if len(p.coords)<100 and len(p.coords) >1]
    print("low contrast filtered: ", len(props_low_contrast_filtered))
    print("high contrast filtered: ", len(props_high_contrast_filtered))
    print("low contrast large: ", len(props_low_contrast_large))
    overlapping= set()
    # for each low contrast region, find the high contrast labels that it contains
    # assume that each high contrast label corresponds to a different cell
    for prop in props_low_contrast_large:
        res = overlaps_region(prop, labels_high_contrast)
        overlapping = overlapping.union(res)
    show_cells_counted = np.zeros_like(image_arr)
    for label in overlapping:
        for coord in props_high_contrast[label].coords:
            show_cells_counted[coord[0], coord[1]] = 3
    for prop in props_low_contrast_filtered:
        for coord in prop.coords:
            show_cells_counted[coord[0], coord[1]] = 1
    for prop in props_low_contrast_large:
        for coord in prop.coords:
            show_cells_counted[coord[0], coord[1]] = 2
    plt.imshow(show_cells_counted)
    plt.show()
    plt.imshow(show_cells_counted)
    plt.savefig("cells_counted.png")
    number_cells = count_cells_simple(props_low_contrast)
    print("%d cells found using naive method" % number_cells)
    print("found %d cells found in total %s"% (len(overlapping)+number_cells, filename))


count_cells_neun("matt/matt_neun_smaller.png") # 2488

# now need to quantify cells in ranges of x pixels from center
# so need to isolate center

