import numpy as np
from scipy.ndimage.morphology import binary_dilation
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import median_filter
from skimage.color import label2rgb
import os
from skimage import morphology


def open_image(filename):
    img = Image.open(filename).convert('L')
    img.load()
    return np.array(img)

def get_injection_site_props(image_arr, pixel_thresh=80, thresh_object=500, thresh_hole=5000):
    markers = np.zeros_like(image_arr)
    markers[image_arr < pixel_thresh] = 1
    print("made markers")
    plt.imshow(markers)
    plt.savefig("markers.png")
    labels = measure.label(markers)
    labels_filtered = morphology.remove_small_objects(labels, thresh_object)
    plt.imshow(labels_filtered)
    plt.savefig("labels_filtered.png")
    labels_final = measure.label(morphology.remove_small_holes(labels_filtered, thresh_hole))
    plt.imshow(labels_final)
    plt.savefig("labels_final.png")
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


class IntensityResults(object):
    def __init__(self):
        self.sums = []
        self.region_intensities = []
        self.intensities_in_masks = []
        self.areas_full = []
        self.areas_with_previous_subtracted = []


    def average_intensity_per_region(self):
        return [int(i)/int(a) for i,a in zip(self.region_intensities, self.areas_with_previous_subtracted)]


def compare_intensities(image_arr, injection_site, iteration_length, iterations_needed):
    injection_site_coords = injection_site.coords
    mask = np.zeros_like(image_arr)
    for x, y in injection_site_coords:
        mask[x,y] = 1
    initial_area = np.sum(mask)
    print(initial_area)
    # problem is that first intensity is entire image- initial mask, need 
    pixels_per_iteration = iteration_length*7 # 7 pixels per micro meter
    masked = np.ma.masked_array(image_arr,mask=~np.array(mask, dtype=bool))
    intensity_of_first_mask = np.sum(masked.compressed())
    mask = binary_dilation(mask, iterations=pixels_per_iteration)
    masked = np.ma.masked_array(image_arr,mask=~mask)
    intensity = np.sum(masked.compressed()) - intensity_of_first_mask # so first intensity will actually be intensity of everything outside mask
    res = IntensityResults()
    for i in range(iterations_needed):
        # intensity in region is total intensity including region - total instensity excluding region
        res.region_intensities.append(intensity)
        intensity = np.sum(masked.compressed())
        res.intensities_in_masks.append(intensity)
        res.areas_full.append(np.sum(mask))
        area = np.sum(mask)
        mask = binary_dilation(mask, iterations=pixels_per_iteration)
        masked = np.ma.masked_array(image_arr,mask=~mask)
        intensity = np.sum(masked.compressed()) - intensity   
        area = np.sum(mask) - area
        res.areas_with_previous_subtracted.append(area)
        if i%20 == 0: 
            print("finished iteration %d" % (i))
    print(res.areas_with_previous_subtracted)
    print(res.region_intensities)
    return res

matt_test_image = 'matt/MAX_Iba1.tif'
image_arr =open_image(matt_test_image)
injection_site = get_injection_site_props(image_arr, 70, 100000, 100000)
intensities = compare_intensities(image_arr, injection_site,1, "matt/MAX_Iba1_max.png")

final = intensities.average_intensity_per_region()



def compare_intensities(image_arr, mask, iteration_length, iterations_needed):
    initial_area = np.sum(mask)
    print(initial_area)
    # problem is that first intensity is entire image- initial mask, need 
    pixels_per_iteration = iteration_length*7 # 7 pixels per micro meter
    masked = np.ma.masked_array(image_arr,mask=~np.array(mask, dtype=bool))
    intensity_of_first_mask = np.sum(masked.compressed())
    mask = binary_dilation(mask, iterations=pixels_per_iteration)
    masked = np.ma.masked_array(image_arr,mask=~mask)
    intensity = np.sum(masked.compressed()) - intensity_of_first_mask # so first intensity will actually be intensity of everything outside mask
    res = IntensityResults()
    for i in range(iterations_needed):
        # intensity in region is total intensity including region - total instensity excluding region
        res.region_intensities.append(intensity)
        intensity = np.sum(masked.compressed())
        res.intensities_in_masks.append(intensity)
        res.areas_full.append(np.sum(mask))
        area = np.sum(mask)
        mask = binary_dilation(mask, iterations=pixels_per_iteration)
        masked = np.ma.masked_array(image_arr,mask=~mask)
        intensity = np.sum(masked.compressed()) - intensity   
        area = np.sum(mask) - area
        res.areas_with_previous_subtracted.append(area)
        if i%20 == 0: 
            print("finished iteration %d" % (i))
    print(res.areas_with_previous_subtracted)
    print(res.region_intensities)
    return res

