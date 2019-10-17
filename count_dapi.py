import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import exposure
from skimage import filters
from skimage import measure
import colorsys
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import grey_erosion, binary_erosion
from skimage import color

def HSVColor(img_arr):
        new_arr = np.zeros_like(img_arr)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                hue = img_arr[i, j, 0]
                sat = img_arr[i, j, 1]
                val = img_arr[i, j, 2]
                r, g, b = colorsys.hsv_to_rgb(np.float(hue/255),np.float(sat/255),np.float(val/255))
                new_arr[i, j, 0] = int(r*255)
                new_arr[i, j, 1] = int(g*255)
                new_arr[i, j, 2] = int(b*255)           
        return new_arr


def isolate_blue(filename):
    # should be a nicer way of doing this but can't find one that works for my python setup
    img_hsv = Image.open(filename).convert('HSV')
    image_arr_hsv = np.array(img_hsv)
    blue_channel = np.zeros_like(image_arr_hsv)
    for i in range(image_arr_hsv.shape[0]):
        for j in range(image_arr_hsv.shape[1]):
            if (image_arr_hsv[i, j, 0] < 270 and image_arr_hsv[i, j, 0] > 160) | (image_arr_hsv[i, j, 0]> 80 and image_arr_hsv[i, j, 0] <90 and image_arr_hsv[i, j, 1] > 210 and image_arr_hsv[i, j, 2] < 55):# and image_arr_hsv[i,j,2] > 150:
                blue_channel[i, j] = image_arr_hsv[i, j]
            else:
                blue_channel[i,j] = np.array([0,0,0])
    return HSVColor(blue_channel)


def label_dapi(array):
    gamma_corrected = exposure.adjust_gamma(array, 0.95, 0.8)
    dapi_filter = lambda x: ((x[2] > 0.7))
    unsharp_masked = filters.unsharp_mask(gamma_corrected, radius=5, amount=20, multichannel=True)
    match_arr = np.zeros((unsharp_masked.shape[0], unsharp_masked.shape[1]))
    for i in range(len(unsharp_masked)):
            for j in range(len(unsharp_masked[i])):
                    if dapi_filter(unsharp_masked[i][j]):
                        unsharp_masked[i][j] = np.array([1, 1, 1], dtype=np.uint8)
                        match_arr[i, j] = 1
    labelled = measure.label(binary_erosion(match_arr))
    labelling_final = measure.label(remove_small_objects(labelled, 15))
    return labelling_final


def overlaps_dapi_region(prop, dapi_labels):
    slice_in_dapi_labels = dapi_labels[prop.slice[0], prop.slice[1]]
    if np.any(slice_in_dapi_labels != 0): # nonzero value corresponds to label!!
        image = np.zeros_like(prop.image, dtype=np.int16)
        print("image shape: ", image.shape)
        #for i in range(len(prop.image)):
        #    for j in range(len(prop.image[i])):
        #        if prop.image[i,j] == True:
        #            image[i,j] = 1
        for i in range(len(slice_in_dapi_labels)):
            for j in range(len(slice_in_dapi_labels[i])):
                if slice_in_dapi_labels[i, j] != 0:
                    #image[i,j] = slice_in_dapi_labels[i, j]
                    #image[i,j] = 2
                    print("slice in dapi labels: %d, i: %d, j: %d" % (slice_in_dapi_labels[i, j], i, j))
        #print(image)
        #plt.imshow(image)
        #plt.show()
        return set(slice_in_dapi_labels[np.where(slice_in_dapi_labels != 0)])
    return set() 



dapi24 = 'maria/count_cell_images/dapi_staining/iba1dapi2.4.jpg'
blue = isolate_blue(dapi24)
labelled_dapi = label_dapi(blue)
merge4 = "maria/count_cells2/Merge-2.4.jpg"
img_hsv = Image.open(merge4).convert('HSV')
image_arr_hsv = np.array(img_hsv)


# script for yellow bit
yellow_channel = np.zeros_like(image_arr_hsv)
yellow_channel_bin = np.array([[0 for i in range(image_arr_hsv.shape[0])]for j in range(image_arr_hsv.shape[1])])
for i in range(image_arr_hsv.shape[0]):
    for j in range(image_arr_hsv.shape[1]):
        if image_arr_hsv[i, j, 0] < 50 and image_arr_hsv[i, j, 0] > 37 and image_arr_hsv[i, j, 2] > 50 and image_arr_hsv[i, j, 1] > 100:
            yellow_channel[i, j] = image_arr_hsv[i, j]
            yellow_channel_bin[i,j] = 1
        else:
            yellow_channel[i,j] = np.array([0,0,0])


yellow_channel_greyscale = np.array(Image.fromarray(HSVColor(yellow_channel)).convert("L"))
ycf = gaussian_filter(yellow_channel_greyscale, 3)
bin_image = np.zeros_like(ycf)
bin_image[ycf>4] = 1

bin_labelled = measure.label(bin_image)
bin_labelled_final = remove_small_objects(bin_labelled, 15)

# now need to co-localise with dapi
props = measure.regionprops(bin_labelled_final, ycf)
overlapping_dapi= set()
for prop in props:
    res = overlaps_dapi_region(prop, labelled_dapi)
    overlapping_dapi = overlapping_dapi.union(res)


# want to draw the dapi regions where yellow has been found
# make an empty array and draw all the dapi labels
dapi_gs= color.rgb2gray(blue)
props_dapi = measure.regionprops(labelled_dapi, dapi_gs)

overlapping = np.zeros_like(dapi_gs)
for label in overlapping_dapi:
    prop = props_dapi[label]
    for coord in prop.coords:
        overlapping[coord[0], coord[1]] = 300



# the above gets very few overlapping- try overlapping just the blue and the yellow
both = np.zeros_like(labelled_dapi)
for i in range(len(labelled_dapi)):
    for j in range(len(labelled_dapi[i])):
        if bin_labelled_final[i, j] != 0:
            both[i][j] = 2
        if labelled_dapi[i,j] != 0:
            both[i][j] = 1

# need to blur the yellow a bit so its overlapping better
# need to include very light blue

# overlapping logic is wrong.... not only are hardly any shown, they don't overlap!!!
def isolate_yellow(yellow_filename):
    img_hsv = Image.open(yellow_filename).convert('HSV')
    image_arr_hsv = np.array(img_hsv)
    yellow_channel = np.zeros_like(image_arr_hsv)
    yellow_channel_bin = np.array([[0 for i in range(image_arr_hsv.shape[0])]for j in range(image_arr_hsv.shape[1])])
    for i in range(image_arr_hsv.shape[0]):
        for j in range(image_arr_hsv.shape[1]):
            if image_arr_hsv[i, j, 0] < 50 and image_arr_hsv[i, j, 0] > 37 and image_arr_hsv[i, j, 2] > 50 and image_arr_hsv[i, j, 1] > 100:
                yellow_channel[i, j] = image_arr_hsv[i, j]
                yellow_channel_bin[i,j] = 1
            else:
                yellow_channel[i,j] = np.array([0,0,0])
    yellow_channel_greyscale = np.array(Image.fromarray(HSVColor(yellow_channel)).convert("L"))
    ycf = gaussian_filter(yellow_channel_greyscale, 4)
    plt.imshow(ycf)
    plt.show()
    bin_image = np.zeros_like(ycf)
    bin_image[ycf>3] = 1
    bin_labelled = measure.label(bin_image)
    bin_labelled_final = remove_small_objects(bin_labelled, 15)
    props = measure.regionprops(bin_labelled_final, ycf)
    return props, bin_labelled_final


def compare_yellow_dapi(yellow_filename, dapi_filename):
    blue = isolate_blue(dapi_filename)
    labelled_dapi = label_dapi(blue)
    yellow_props, bin_labelled_final = isolate_yellow(yellow_filename)
    # now need to co-localise with dapi
    overlapping_dapi= set()
    yellow_with_dapi = 0
    overlaps = []
    for prop in yellow_props:
        res = overlaps_dapi_region(prop, labelled_dapi)
        if res != set():
            yellow_with_dapi += 1
            overlaps.append(prop)
        overlapping_dapi = overlapping_dapi.union(res)
    print("yellow_with_dapi: %d" % yellow_with_dapi)
    # want to draw the dapi regions where yellow has been found
    # make an empty array and draw all the dapi labels
    dapi_gs= color.rgb2gray(blue)
    props_dapi = measure.regionprops(labelled_dapi, dapi_gs)
    # map over yellow to show which have been counted
    for label in overlapping_dapi:
        prop = props_dapi[label-1]
        for coord in prop.coords:
            bin_labelled_final[coord[0], coord[1]] = 300
    yellow_selected = np.zeros_like(dapi_gs)
    for p in yellow_props:
        for coord in p.coords:
            yellow_selected[coord[0], coord[1]]=1
    for prop in props_dapi:
        for coord in prop.coords:
            yellow_selected[coord[0], coord[1]] = 2
    plt.imshow(yellow_selected)
    plt.show()
    return bin_labelled_final, overlapping_dapi, props_dapi



overlapping_image, overlapping_labels, props_dapi = compare_yellow_dapi(merge4, dapi24)

# looking at each individually, there is an overlap- so something is wrong with the visualisation


# need to make a test simple image to fix overlapping error

basic_array1 = np.zeros((20,20))
basic_array1[3:8,13:18] = 1
basic_array1[13:18,3:8] = 2
basic_array2 = np.zeros((20,20))
basic_array2[6:9,12:15] = 3
basic_array2[6:9,6:9] = 4
basic_array2[12:15,12:15] = 5
basic_array2[10:14,6:9] = 6

both = np.zeros_like(basic_array1)
for i in range(len(basic_array1)):
    for j in range(len(basic_array1[i])):
        if basic_array1[i, j] != 0:
            both[i][j] = 2
        if basic_array2[i,j] != 0:
            both[i][j] = 1

labelled1 = measure.label(basic_array1)
labelled = measure.label(basic_array2)
props = measure.regionprops(labelled, basic_array2)
overlapping_dapi= set()
for prop in props:
    res = overlaps_dapi_region(prop, labelled1)
    overlapping_dapi = overlapping_dapi.union(res)

overlapping = np.zeros_like(labelled1)
for label in overlapping_dapi:
    prop = props[label]
    for coord in prop.coords:
        overlapping[coord[0], coord[1]] = 2

# this small exapmle works!!!







