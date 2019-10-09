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
from scipy.ndimage.morphology import grey_erosion, binary_erosion


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
    img_hsv = Image.open(filename).convert('HSV')
    image_arr_hsv = np.array(img_hsv)
    blue_channel = np.zeros_like(image_arr_hsv)
    for i in range(image_arr_hsv.shape[0]):
        for j in range(image_arr_hsv.shape[1]):
            if image_arr_hsv[i, j, 0] < 270 and image_arr_hsv[i, j, 0] > 160:# and image_arr_hsv[i,j,2] > 150:
                blue_channel[i, j] = image_arr_hsv[i, j]
            else:
                blue_channel[i,j] = np.array([0,0,0])
    return blue_channel


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
    #labelling_final = measure.label(remove_small_objects(grey_erosion(labelled, size=2), 15))
    labelling_final = measure.label(remove_small_objects(labelled, 15))
    return labelling_final


dapi24 = 'count_cell_images/dapi_standing/iba1dapi2.4.jpg'
dapi = HSVColor(isolate_blue(dapi24))
labeled = label_dapi(dapi)

# some connections that we don't want, try binary erosion


