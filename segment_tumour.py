from scipy.ndimage.morphology import binary_dilation
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.filters import gaussian, median
from skimage import measure
from skimage import exposure
from scipy.ndimage import median_filter
from skimage.color import label2rgb
import os
import skimage
from skimage.morphology import skeletonize
from skimage.morphology import remove_small_objects, remove_small_holes

def open_image(filename):
    img = Image.open(filename)
    img.load()
    return np.array(img)

image_folder = "/Users/user/Documents/image_analysis/paul/processed_images/"
U87_GO_17_4a = "U87-GO-17-4a/"
tumour_image = "U87-GO-17_4a_x20_all.jpg"
go_image = "U87-GO-17_4a_x20_BF"

tumour_image_arr = open_image(image_folder + U87_GO_17_4a + tumour_image)

from skimage import feature
from skimage import color

# nope
edges = feature.canny(color.rgb2gray(tumour_image_arr), sigma=4)

tumour_gs = color.rgb2gray(tumour_image_arr)

segmented = np.zeros_like(tumour_gs)
segmented[tumour_gs > 0.16] =1

edges = feature.canny(segmented, sigma=3)
# insanely slow

blobs_log = feature.blob_log(tumour_gs, max_sigma=30, num_sigma=10, threshold=.1)

fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True)

ax.imshow(tumour_gs)

for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, color="b", linewidth=2, fill=False)
    ax.add_patch(c)
ax.set_axis_off()


# try blurring
from skimage.filters import gaussian

from skimage import exposure

contrast_increased = exposure.adjust_log(tumour_gs, gain=8)
plt.imshow(contrast_increased)
plt.show()

tumour_gs_gaussian = gaussian(contrast_increased, sigma=20)
plt.imshow(tumour_gs_gaussian)
plt.show()

segmented = np.zeros_like(tumour_gs)
segmented[tumour_gs_gaussian > 1.5] =1
plt.imshow(segmented)
plt.show()

filter = lambda x: ((20 < x[0] < 40) & (15 < x[1] <35) & (65 < x[2]<95) | (x[0] < 100) & (120 < x[2]<255)| (130 < x[0] < 187) & (150 < x[1] <187) & (192 < x[2]<255) | (100 < x[0] < 120) & (140 < x[1] <165) & (190 < x[2]<255)  | (200 < x[0] < 210) & (205 < x[1] <220) & (220 < x[2]<235))
bitmap = (((image_arr[:,:,0] > 20) & (image_arr[:,:,0] < 40)) & ((image_arr[:,:,1] > 15) & (image_arr[:,:,1] < 35)) & ((image_arr[:,:,2] > 65) & (image_arr[:,:,2] < 95))) | \
        ((image_arr[:,:,0] < 100) & ((image_arr[:,:,1] > 15) & (image_arr[:,:,1] < 35)) & ((image_arr[:,:,2] > 120) & (image_arr[:,:,2] < 225))) | \
        (((image_arr[:,:,0] > 100) & (image_arr[:,:,0] < 120)) & ((image_arr[:,:,1] > 140) & (image_arr[:,:,1] < 165)) & ((image_arr[:,:,2] > 190) & (image_arr[:,:,2] < 255))) |\
        (((image_arr[:,:,0] > 200) & (image_arr[:,:,0] < 210)) & ((image_arr[:,:,1] > 205) & (image_arr[:,:,1] < 220)) & ((image_arr[:,:,2] > 220) & (image_arr[:,:,2] < 235))) 

image_arr[bitmap==True] = [255, 255, 0]

def isolate_blue(filename):
    # should be a nicer way of doing this but can't find one that works for my python setup
    img_hsv = Image.open(filename).convert('HSV')
    image_arr_hsv = np.array(img_hsv)
    blue_channel = np.zeros_like(image_arr_hsv)
    # want variation in blue for 
    #bitmap = ((image_arr_hsv[:,:,0] < 270) & (image_arr_hsv[:,:,0] > 160))
    #blue_channel[bitmap==True]
    for i in range(image_arr_hsv.shape[0]):
        for j in range(image_arr_hsv.shape[1]):
            if (image_arr_hsv[i, j, 0] < 270 and image_arr_hsv[i, j, 0] > 160) | (image_arr_hsv[i, j, 0]> 80 and image_arr_hsv[i, j, 0] <90 and image_arr_hsv[i, j, 1] > 210 and image_arr_hsv[i, j, 2] < 55):
                blue_channel[i, j] = image_arr_hsv[i, j]
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

# this is incredibly slow for this image..... or maybe cos i'm caning my ram with the other analysis?
blue = isolate_blue(image_folder + U87_GO_17_4a + tumour_image)
labelled_dapi = label_dapi(blue)


# should probably do by green- then median filter + edge detection?
green_tumour = np.zeros_like(tumour_gs)
green_bitmap = (((tumour_image_arr[:,:,0] < 100) & (tumour_image_arr[:,:,1] > 100)& (tumour_image_arr[:,:,2] < 100))| ((tumour_image_arr[:,:,0] < 50) & (tumour_image_arr[:,:,1] > 50)& (tumour_image_arr[:,:,2] < 50)))
green_tumour[green_bitmap == True] = 1


# try massively blurring- think i'll get same problem as previous approach

tumour_green_gaussian = gaussian(green_tumour, sigma=20)
plt.imshow(tumour_green_gaussian)
plt.show()

#edges = feature.canny(tumour_green_gaussian)
#plt.imshow(edges)
#plt.show()

markers = np.zeros_like(tumour_green_gaussian)
markers[tumour_green_gaussian>0.15] = 1
plt.imshow(markers)
plt.show()
labels = measure.label(markers)
plt.imshow(labels)
plt.show()

labels_filtered = skimage.morphology.remove_small_objects(labels, 70000)
plt.imshow(labels_filtered)
plt.show()

# then need to blur again to join....
labels_filtered_bin = np.zeros_like(labels_filtered)
labels_filtered_bin[labels_filtered>0] = 1
labels_filtered_blurred = gaussian(labels_filtered_bin, sigma=150)
plt.imshow(labels_filtered_blurred)
plt.show()
labels_filtered_blurred*=1E20
final_bin = np.zeros_like(labels_filtered_blurred)
final_bin[labels_filtered_blurred >7.5] = 1
from skimage.morphology import skeletonize
final_bin_skeleton = skeletonize(final_bin)
#from skimage.morphology import medial_axis
#skel, distance = medial_axis(final_bin, return_distance=True)
#dist_on_skel = distance * skel
from matplotlib import path
labelled_final = measure.label(final_bin_skeleton)
props = measure.regionprops(labelled_final)
p = path.Path(props[1].coords)
#from skimage.morphology import thin


def locate_tumour_edge_not_working(filename, marker_thresh=1):
    tumour_image_arr = open_image(filename)
    tumour_gs = color.rgb2gray(tumour_image_arr)
    green_tumour = np.zeros_like(tumour_gs)
    green_bitmap = (((tumour_image_arr[:,:,0] < 100) & (tumour_image_arr[:,:,1] > 100)& (tumour_image_arr[:,:,2] < 100))| ((tumour_image_arr[:,:,0] < 50) & (tumour_image_arr[:,:,1] > 50)& (tumour_image_arr[:,:,2] < 50)))
    green_tumour[green_bitmap == True] = 1
    tumour_green_gaussian = gaussian(green_tumour, sigma=20)
    plt.imshow(tumour_green_gaussian)
    plt.show()
    markers = np.zeros_like(tumour_green_gaussian)
    markers[tumour_green_gaussian>marker_thresh] = 1
    plt.imshow(markers)
    plt.show()
    labels = measure.label(markers)
    labels_filtered = skimage.morphology.remove_small_objects(labels, 70000)
    plt.imshow(labels_filtered)
    plt.show()    
    # then need to blur again to join....
    labels_filtered_bin = np.zeros_like(labels_filtered)
    labels_filtered_bin[labels_filtered>0] = 1
    labels_filtered_blurred = gaussian(labels_filtered_bin, sigma=150)
    labels_filtered_blurred*=1E20
    plt.imshow(labels_filtered_blurred)
    plt.show()
    final_bin = np.zeros_like(labels_filtered_blurred)
    final_bin[labels_filtered_blurred >7.5] = 1
    plt.imshow(final_bin)
    plt.show()
    final_bin_skeleton = skeletonize(final_bin)
    labelled_final = measure.label(final_bin_skeleton)
    return labelled_final


tumour_boundary = locate_tumour_edge_not_working(image_folder + U87_GO_17_4a + tumour_image)
props = measure.regionprops(labelled_final)
p = path.Path(props[1].coords)


# now need to try for whole brain image...
whole_brain_image = "U87-GO-17_4a_x4_all.jpg"
# finds the left brain ventricle
brain_boundary = locate_tumour_edge_not_working(image_folder + U87_GO_17_4a + whole_brain_image, 0.013)


brain_image_arr = open_image(image_folder + U87_GO_17_4a + whole_brain_image)
tumour_gs = color.rgb2gray(tumour_image_arr)
from skimage.filters import unsharp_mask
result_1 = unsharp_mask(brain_image_arr, radius=30, amount=2)

# try just increasing contrast

#contrast_increased = exposure.adjust_log(brain_image_arr, gain=1.5)
#plt.imshow(contrast_increased)
#plt.show()

contrast_increased = exposure.adjust_gamma(brain_image_arr, gain=10, gamma=4)
plt.imshow(contrast_increased)
plt.show()

tumour_gaussian = gaussian(contrast_increased, sigma=20)
plt.imshow(tumour_gaussian)
plt.show()

# try increasing contrast and blurring again
contrast_increased2 = exposure.adjust_gamma(tumour_gaussian, gain=30, gamma=4)
plt.imshow(contrast_increased2)
plt.show()


# try edge detection on this.... doesn't work yet again....
#edges = feature.canny(color.rgb2gray(tumour_gaussian2), sigma=4)
tumour_gs = color.rgb2gray(contrast_increased2)
markers = np.zeros_like(color.rgb2gray(tumour_gs))
markers[tumour_gs > 0.015] = 1

from skimage.morphology import remove_small_holes
labelled = measure.label(markers)
def get_largest_connected_component(segmentation):
    labels = measure.label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return remove_small_holes(measure.label(largest_cc), 500000)

largest_label = get_largest_connected_component(markers)
props = measure.regionprops(largest_label)

brightfield_brain = "U87-GO-17_4a_x4_BF.jpg"
brain_image_arr = open_image(image_folder + U87_GO_17_4a + brightfield_brain)
# now try getting outline of brain
contrast_increased = color.rgb2gray(exposure.adjust_gamma(brain_image_arr, gain=50, gamma=8))
plt.imshow(contrast_increased)
plt.show()
mask_brain = np.zeros_like(contrast_increased)
mask_brain[contrast_increased < 0.7] = 1
mask_brain = remove_small_holes(mask_brain.astype(int), 10000)
labelled_brain = measure.label(mask_brain)
labelled_brain = measure.label(remove_small_objects(labelled_brain, 20000))
brain_contour = measure.find_contours(labelled_brain, 0.5)

contour_image_brain = np.zeros_like(mask_brain)
for i in range(len(brain_contour)):
    for j in range(len(brain_contour[i])):
        contour_image_brain[int(brain_contour[i][j][0]), int(brain_contour[i][j][1])] = 1

contour_image_brain = binary_dilation(binary_dilation(binary_dilation(contour_image_brain)))

# to isolate tumour- can do inverse mask
# try alpha composite to demonstrate what i've found.....
#tumour_image = tumour_image.convert("RGB").putalpha(1)
#tumour_image = Image.fromarray(~largest_label)
#brain_brightfield = Image.open(image_folder + U87_GO_17_4a + brightfield_brain).convert("L")
#brain_brightfield.putalpha(Image.fromarray(~largest_label).convert("L"))
#alpha_composited = Image.alpha_composite(tumour_image, brain_brightfield)

# alpha being annoying.... just try with mask
brightfield_brain = "U87-GO-17_4a_x4_BF.jpg"
brain_brightfield = Image.open(image_folder + U87_GO_17_4a + brightfield_brain).convert("L")
from skimage.filters import median
median_filtered = median(np.array(brain_brightfield))
masked = np.ma.masked_array(np.array(brain_brightfield)[:2300, :3560], mask = ~largest_label)
# not what i want to be doing.... threshold to get go, then median filter

bb_arr = np.array(brain_brightfield)[:2300, :3560]
thresholded = np.zeros_like(bb_arr)
for i in range(len(bb_arr)):
    for j in range(len(bb_arr[i])):
            if bb_arr[i, j] < 100:
                thresholded[i, j] = bb_arr[i, j]

med_thresholded = median(thresholded)
masked = np.ma.masked_array(thresholded, mask = ~largest_label)

bb_arr_rgb = np.array(Image.open(image_folder + U87_GO_17_4a + brightfield_brain))
for i in range(len(masked)):
    for j in range(len(masked[i])):
        if masked[i, j] != 0:
            bb_arr_rgb[i,j] = np.array([255-masked[i,j],0,0])


def locate_tumour(image_filename, marker_threshold= 0.15):
    brain_image_arr = open_image(image_filename)
    contrast_increased = exposure.adjust_gamma(brain_image_arr, gain=10, gamma=4)
    tumour_gaussian = gaussian(contrast_increased, sigma=20)
    # increase contrast and blurring again
    contrast_increased2 = exposure.adjust_gamma(tumour_gaussian, gain=30, gamma=4)
    # try edge detection on this.... doesn't work yet again....
    tumour_gs = color.rgb2gray(contrast_increased2)
    plt.imshow(tumour_gs)
    plt.show()
    markers = np.zeros_like(color.rgb2gray(tumour_gs))
    markers[tumour_gs > marker_threshold] = 1
    plt.imshow(markers)
    plt.show()
    largest_label = get_largest_connected_component(markers)
    return largest_label

tumour = locate_tumour(image_folder + U87_GO_17_4a + whole_brain_image, 0.1)
masked = np.ma.masked_array(thresholded, mask = ~largest_label)

tumour_graphene = tumour.astype(int)*thresholded
tumour_graphene_bin = np.zeros_like(tumour_graphene)
tumour_graphene_bin[tumour_graphene>0] = 1

window = np.ones((5,5))
#from scipy.signal import fftconvolve
#from matplotlib import cm, colors
windowed_average = fftconvolve(tumour_graphene_bin, window)
norm = colors.Normalize(vmin=np.min(windowed_average), vmax=np.max(windowed_average))
m = cm.ScalarMappable(norm=norm, cmap=cm.hot)
bb_arr_rgb = np.array(Image.open(image_folder + U87_GO_17_4a + brightfield_brain))
for i in range(len(bb_arr_rgb)):
    for j in range(len(bb_arr_rgb[i])):
        if windowed_average[i, j] > 0.1:# really really slow like this....
            converted = m.to_rgba(windowed_average[i,j])[:-1]
            bb_arr_rgb[i,j] = [converted[0]*255,converted[1]*255,converted[2]*255]


# https://stackoverflow.com/questions/31877353/overlay-an-image-segmentation-with-numpy-and-matplotlib
window = np.ones((10,10))
windowed_average = fftconvolve(tumour_graphene_bin, window)
norm = colors.Normalize(vmin=np.min(windowed_average), vmax=np.max(windowed_average))
m = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
average_colourmapped = m.to_rgba(windowed_average)
#plt.imshow(bb_arr_rgb)
#plt.imshow(average_colourmapped, alpha=0.5)
#plt.show()
# then tumour outline and brain contour (half image)

mask_3d = np.zeros_like(average_colourmapped)
for i in range(len(tumour)):
    for j in range(len(tumour[i])):
        if tumour[i,j] == 1:
            if average_colourmapped[i,j, 2] > 0.525 and average_colourmapped[i,j, 2] < 0.53:
                #print("yes")
                pass
            else:
                mask_3d[i,j] = average_colourmapped[i,j]

tumour_contour = measure.find_contours(tumour, 0.5)

contour_image = np.zeros((mask_3d.shape[0],mask_3d.shape[1]))
for i in range(len(tumour_contour[0])):
    contour_image[int(tumour_contour[0][i][0]), int(tumour_contour[0][i][1])] = 1

contour_image = binary_dilation(binary_dilation(binary_dilation(contour_image)))

for i in range(len(contour_image)):
    for j in range(len(contour_image[i])):
        if contour_image[i, j] == 1:
            mask_3d[i,j] = np.array([0, 255,0,1])
        if contour_image_brain[i,j] == 1:
            mask_3d[i,j] = np.array([255,0,0,1])


#plt.imshow(bb_arr_rgb[:, 1750:])
#plt.imshow(mask_3d[:, 1750:], alpha=0.5)
#plt.show()

# want bar and image to be same size: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
plt.figure()
ax = plt.gca()
im = ax.imshow(mask_3d)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(m, cax=cax)

# need to try nonlinear colourbar
# https://stackoverflow.com/questions/22521382/nonlinear-colormap-matplotlib


# quantify area its travelled

# code- very bright colour, for very dense graphene, much less bright colour for less dense
# pixels of grey per unit area.
# what distance does 50% travel?

# for density, convolve with array of ones. can't convolve with masked array, so multiply by tumour

# cut image in half, outline of brain in one colour, tumour outline in another colour
# how many sections is GO present in? give rough estimate of volume GO has moved through

# increase contrasts in density map - make redder where there's more GO and add colour bar


