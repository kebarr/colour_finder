from scipy.ndimage import median_filter
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import skimage
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage import measure

def count_yellow(filename):
    img = Image.open(filename)
    img.load()
    image_arr = np.array(img)
    filter_match = lambda x: ((150<x[0]) & (115 < x[1]) & (35<x[2]<60))
    match_arr = np.zeros((image_arr.shape[0], image_arr.shape[1]))
    for i in range(len(image_arr)):
            for j in range(len(image_arr[i])):
                    if filter_match(image_arr[i][j]):
                            match_arr[i, j] = 1
    filtered = median_filter(match_arr, size=3) # orignally done with size = 3
    lbl = measure.label(filtered)
    res = count_cells(lbl, match_arr)
    return res


def get_gs_array(filename):
    img_gs = Image.open(filename).convert('L')
    img_gs.load()
    return np.array(img_gs)

def get_segmented_image(array, marker_lower, marker_upper):
    markers = np.zeros_like(array)
    markers[array < marker_lower] = 1
    markers[array > marker_upper] = 2
    elevation_map = sobel(array)
    return watershed(elevation_map, markers)


def count_cells(to_count, original_array):
    labels = measure.label(to_count)
    props = measure.regionprops(labels, original_array)
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


filename_green = '../maria/green.png'
filename_yellow = '../maria/green_red_yellow.png'
cells_yellow = count_yellow(filename_yellow)
green_arr = get_gs_array(filename_green)
green_segmented = get_segmented_image(green_arr, 30, 60)
cells_green = count_cells(green_segmented, green_arr)
final = 100*cells_yellow/cells_green