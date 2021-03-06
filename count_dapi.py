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
            if (image_arr_hsv[i, j, 0] < 270 and image_arr_hsv[i, j, 0] > 160) | (image_arr_hsv[i, j, 0]> 80 and image_arr_hsv[i, j, 0] <90 and image_arr_hsv[i, j, 1] > 210 and image_arr_hsv[i, j, 2] < 55):
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
        return set(slice_in_dapi_labels[np.where(slice_in_dapi_labels != 0)])
    return set() 



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
    dapi_gs= color.rgb2gray(blue)
    props_dapi = measure.regionprops(labelled_dapi, dapi_gs)
    # map over yellow to show which have been counted
    yellow_selected = np.zeros_like(dapi_gs)
    for p in yellow_props:
        for coord in p.coords:
            yellow_selected[coord[0], coord[1]]=1
    for prop in props_dapi:
        for coord in prop.coords:
            yellow_selected[coord[0], coord[1]] = 2
    plt.imshow(yellow_selected)
    filename = yellow_filename.split(".jpg")[0] + "_dapi_yellow_combined.png"
    plt.savefig(filename)
    print("located %d cells in %s" %(len(overlapping_dapi), yellow_filename))


dapi24 = 'maria/count_cell_images/dapi_staining/iba1dapi2.4.jpg'
merge4 = "maria/count_cells2/Merge-2.4.jpg"
overlapping_image, overlapping_labels = compare_yellow_dapi(merge4, dapi24)


# now try and run for all....
maria_images = os.listdir("maria/count_cell_images")
merge_images = sorted([im for im in maria_images if im.startswith('Merge') and im.endswith('-1.jpg')])
maria_images2 = os.listdir("maria/count_cells2")
merge_images2 = sorted([im for im in maria_images2 if im.startswith('Merge') and im.endswith('.jpg')])
merge_final = sorted(merge_images+merge_images2)

dapi_images = os.listdir("maria/count_cell_images/dapi_staining")
dapi_images = sorted([im for im in dapi_images if im.endswith('.jpg')])

for di, mi in zip(dapi_images, merge_final):
    di = "maria/count_cell_images/dapi_staining/" + di
    mi = "maria/count_cell_images/" + mi
    compare_yellow_dapi(mi, di)
#found 264 green cells in maria/count_cells2/IBA1-1.1.jpeg
#found 505 green cells in ../maria/count_cell_images/IBA1-1.2.jpg
#found 325 green cells in ../maria/count_cell_images/IBA1-1.3.jpg
#found 442 green cells in ../maria/count_cell_images/IBA1-2.1.jpg
#found 514 green cells in ../maria/count_cell_images/IBA1-2.2.jpg
#found 451 green cells in ../maria/count_cell_images/IBA1-2.3.jpg
# found 449 green cells in maria/count_cells2/IBA1-2.4.jpeg
#found 394 green cells in ../maria/count_cell_images/IBA1-3.1.jpg
#found 298 green cells in ../maria/count_cell_images/IBA1-3.2.jpg
#found 323 green cells in ../maria/count_cell_images/IBA1-3.3.jpg
# output:
#yellow_with_dapi: 113 - 42%
#located 343 cells in maria/count_cell_images/Merge-1.1.jpg - 67%
#yellow_with_dapi: 157 - 31%
#located 420 cells in maria/count_cell_images/Merge-1.2-1.jpg - more!!
#yellow_with_dapi: 117 - 36%
#located 348 cells in maria/count_cell_images/Merge-1.3-1.jpg
#yellow_with_dapi: 114 - 26%
#located 211 cells in maria/count_cell_images/Merge-2.1.jpg
#yellow_with_dapi: 108 - 21%
#located 239 cells in maria/count_cell_images/Merge-2.2.jpg
#yellow_with_dapi: 103 -23%
#located 223 cells in maria/count_cell_images/Merge-2.3-1.jpg
#yellow_with_dapi: 73 - 16%
#located 116 cells in maria/count_cell_images/Merge-2.4.jpg
#yellow_with_dapi: 83 - 21%
#located 198 cells in maria/count_cell_images/Merge-3.1-1.jpg
#yellow_with_dapi: 103 - 34%
#located 209 cells in maria/count_cell_images/Merge-3.2-1.jpg
#yellow_with_dapi: 130 - 40%
#located 388 cells in maria/count_cell_images/Merge-3.3-1.jpg

