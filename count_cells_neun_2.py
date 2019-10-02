import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_dilation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skimage
from skimage import measure
from skimage.color import label2rgb
import os
from skimage import morphology
from skimage import exposure


img = Image.open("matt/matt_neun_smaller.png").convert("L")
img.load()
image_arr = np.array(img)
gamma_corrected = exposure.adjust_gamma(image_arr, 8)

# Logarithmic
logarithmic_corrected = exposure.adjust_log(image_arr, 10)
plt.imshow(logarithmic_corrected)
plt.show()
plt.imshow(gamma_corrected)
plt.show()

# high gamma corrected could be suitable
# really high (>100) may enable us to count cells in central regio

gamma_corrected = exposure.adjust_gamma(image_arr, 8)

binary = np.zeros_like(gamma_corrected)
binary = gamma_corrected >50
plt.imshow(binary)
plt.show()


labels = measure.label(binary)
plt.imshow(labels)
plt.show()

# might need to combine higher and lower contrast image.....
# then use similar approach to with marias dapi image
# lower contrast will have more amalgamated blobs, need to separate them based on high contrast image

def get_labelled_array(image_arr, gamma, binary_threshold):
    gamma_corrected = exposure.adjust_gamma(image_arr, gamma)
    binary = np.zeros_like(gamma_corrected)
    binary = gamma_corrected > binary_threshold
    plt.imshow(binary)
    plt.show()
    labels = measure.label(binary)
    return labels

# can't reliably separate cells against really bright background

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
    labels_high_contrast = get_labelled_array(image_arr, 100, 200)
    props_low_contrast = measure.regionprops(labels, image_arr) 
    props_high_contrast = measure.regionprops(labels, image_arr)
    # large low contrast labels are likely to be multiple cells 
    props_low_contrast_large = [p for p in props_low_contrast if p.major_axis_length>10]
    props_low_contrast_filtered = [p for p in props_low_contrast if p.major_axis_length<=10 and p.major_axis_length >2]
    print("low contrast filtered: ", len(props_high_contrast_filtered))
    print("low contrast large: ", len(props_low_contrast_large))
    overlapping= set()
    # for each low contrast region, find the high contrast labels that it contains
    # assume that each high contrast label corresponds to a different cell
    for prop in props_low_contrast_large:
        res = overlaps_region(prop, labels_high_contrast)
        overlapping = overlapping.union(res)
    number_cells = count_cells_simple(props_low_contrast)
    print("%d cells found using naive method" % number_large_cells)
    print("found %d cells found in total %s"% (len(overlapping)+number_cells, filename))



# looks potentially useful... https://clickpoints.readthedocs.io/en/latest/examples/example_plantroot.html
#https://www.hackevolve.com/counting-bricks/