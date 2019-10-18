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
from skimage import feature

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

props = measure.regionprops(labels, image_arr)
show_cells_counted = np.zeros_like(image_arr)
for prop in props_low_contrast_filtered:
        for coord in prop.coords:
            show_cells_counted[coord[0], coord[1]] = 20

# might need to combine higher and lower contrast image.....
# then use similar approach to with marias dapi image
# lower contrast will have more amalgamated blobs, need to separate them based on high contrast image


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
# looks potentially useful... https://clickpoints.readthedocs.io/en/latest/examples/example_plantroot.html
#https://www.hackevolve.com/counting-bricks/\


# try this....
http://scipy-lectures.org/advanced/image_processing/auto_examples/plot_spectral_clustering.html#sphx-glr-advanced-image-processing-auto-examples-plot-spectral-clustering-py
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

labels = spectral_clustering(image_arr)
mask = image_arr.astype(bool)
image_arr = image_arr.astype(float)
image_arr += 1 + 0.2 * np.random.randn(*image_arr.shape)

graph = image.img_to_graph(image_arr, mask=mask)

graph.data = np.exp(-graph.data / graph.data.std())
# this line is incredibly slow- prohibitively slow.....
labels = spectral_clustering(graph, eigen_solver='arpack')
label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

# maybe try splitting image and doing it on smaller versions
# 111 by 111 would give 5
size = 111
image_tiles = []
for i in range(5):
    for j in range(5):
        tile = image_arr[i*111:i*111+111, j*111:j*111+111]
        image_tiles.append(tile)

tile1 = image_tiles[0]
mask = tile1.astype(bool)
graph = image.img_to_graph(tile1, mask=mask)

graph.data = np.exp(-graph.data / graph.data.std())
# this line is incredibly slow- prohibitively slow.....
labels = spectral_clustering(graph, n_clusters = 200, eigen_solver='arpack')
label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

# ha no where near, still not when i increase number of clusters

# try https://scikit-image.org/docs/stable/auto_examples/edges/plot_ridge_filter.html#sphx-glr-auto-examples-edges-plot-ridge-filter-py
from skimage.filters import meijering, sato, frangi, hessian


# hopeful.....
plt.imshow(hessian(image_arr, black_ridges=True))
plt.axis('off')
plt.show() 

# nope:
plt.imshow(meijering(image_arr, black_ridges=False))
plt.axis('off')
plt.show() 

# possible
plt.imshow(sato(image_arr, black_ridges=True))
plt.axis('off')
plt.show() 

# also possible
plt.imshow(frangi(image_arr, black_ridges=True))
plt.axis('off')
plt.show() 

# try segmentation with frangi and sato and compare results
frangi_im = frangi(image_arr, black_ridges=True)
sato_im = sato(image_arr, black_ridges=True)
edges_frangi = feature.canny(frangi_im, sigma=0.2, low_threshold=0.07, \
                      high_threshold=0.18)
plt.imshow(edges_frangi, cmap='gray')

edges_sato = feature.canny(sato_im, sigma=0.2, low_threshold=0.07, \
                      high_threshold=0.18)
plt.imshow(edges_sato, cmap='gray')

# edge detection probably isn't the right approach....

# increase contrast then label....
gamma_corrected_sato = exposure.adjust_gamma(sato_im, 1)
gamma_corrected_frangi = exposure.adjust_gamma(frangi_im, 8)

frangi_im = frangi(image_arr, black_ridges=True)
sato_im = sato(image_arr, black_ridges=True)
# need to mask out center before contrast adjustment. 
for i in range(240,325):
    for j in range(214, 295):
        sato_im[j, i] = 0

# takes out too much outside of center
sato_im[sato_im >20] = 0

gamma_corrected_sato = exposure.adjust_gamma(sato_im, 1)
plt.imshow(gamma_corrected_sato)
plt.show()

# then try labelling
gamma_corrected_sato = exposure.adjust_gamma(sato_im, 1)
labels = measure.label(gamma_corrected_sato)
plt.imshow(labels)
plt.show()
# it labels background and leaves bits we want as foreground
gamma_corrected_sato[np.where(gamma_corrected_sato==0)] = 60
plt.imshow(gamma_corrected_sato)
plt.show()

# actually try logarithmic correction

logarithmic_corrected_sato = exposure.adjust_log(sato_im, 10)
logarithmic_corrected_frangi = exposure.adjust_log(frangi_im, 5)

# log correct makes the contrast higher

logarithmic_corrected_sato = -logarithmic_corrected_sato + 60

logarithmic_corrected_frangi = exposure.adjust_log(frangi_im, 5)
logarithmic_corrected_frangi = -logarithmic_corrected_frangi + 60

# sato looks much better
label_frangi = measure.label(logarithmic_corrected_frangi)
label_sato = measure.label(logarithmic_corrected_sato)

# straight labelling didn't work.... don't need to do - log correction if we threshold


def get_segmented_image(array, marker_lower, marker_upper):
    markers = np.zeros_like(array)
    markers[array < marker_lower] = 2
    markers[array > marker_upper] = 1
    plt.imshow(markers)
    plt.show()
    elevation_map = sobel(array)
    ws = watershed(elevation_map, markers)
    return ws

logarithmic_corrected_sato = exposure.adjust_log(sato_im, 7)
sato_segmented = get_segmented_image(sato_im, 4, 20)

# threshold 20 to above works really well, but still misses some, (log threshold 10)

# think we can just binarise
sato_thresholded = np.zeros_like(sato_im)
sato_thresholded[logarithmic_corrected_sato<3] = 1

sato_labelled = measure.label(sato_thresholded) # gives 1119 cells
plt.imshow(sato_labelled)
plt.show()

# 50% as many as other approach - check way i've totalled them....
# compare output images - original approach varying contrast is better.



