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
        res.areas.append(np.sum(mask)-area)
        area = np.sum(mask)
    return res


image_arr =open_image(matt_test_image)
injection_site = get_injection_site_props(image_arr)

