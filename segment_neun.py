import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image
from skimage import exposure
from skimage.filters import rank_order

def segment_neun(data):
    lowpass = ndimage.gaussian_filter(data, 4)
    labels = data - lowpass
    mask = labels >= 1
    label_values = np.unique(labels)
    labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
    rescaled = exposure.rescale_intensity(labels, out_range=(0, 255))
    markers = np.zeros_like(rescaled)
    markers[rescaled>200] = 1
    return markers

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


def compare_intensities(image_arr, injection_site, pixels_per_iteration, iterations_needed=200):
    injection_site_coords = injection_site.coords
    mask = np.zeros_like(image_arr)
    print(len(injection_site_coords))
    for x, y in injection_site_coords:
        mask[x,y] = 1
    print('made mask ', np.sum(mask))
    # smallest distance outwards
    #iterations_needed = int(np.min(np.array([np.abs(injection_site.bbox[0]- image_arr.shape[0]), np.abs(injection_site.bbox[1]- image_arr.shape[0]), np.abs(injection_site.bbox[2]- image_arr.shape[1]), np.abs(injection_site.bbox[3]- image_arr.shape[1])]))/pixels_per_iteration)
    intensity = np.sum(image_arr)# so first intensity will actually be intensity of everything outside mask
    res = IntensityResults()
    area = np.sum(mask)
    print(pixels_per_iteration)
    for i in range(iterations_needed):
        masked = mask*image_arr
        # intensity in region is total intensity including region - total instensity excluding region
        res.region_intensities.append(intensity-np.sum(masked))
        intensity = np.sum(masked)
        res.sums.append(intensity)
        mask = binary_dilation(mask, iterations=pixels_per_iteration)
        res.areas.append(np.sum(mask)-area)
        area = np.sum(mask)
        print("iteration: %d" % i)
    plt.imshow(mask)
    plt.show()
    return res



