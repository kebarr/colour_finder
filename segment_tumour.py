from scipy.ndimage.morphology import binary_dilation
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.filters import gaussian, median
from skimage import measure, exposure, color
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve
import os
import skimage
from skimage.morphology import remove_small_objects, remove_small_holes
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def open_image(filename):
    img = Image.open(filename)
    img.load()
    return np.array(img)

def get_largest_connected_component(segmentation):
    labels = measure.label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return remove_small_holes(measure.label(largest_cc), 500000)


class SegmentBrainImage(object):
    def __init__(self, brightfield_filename, dapi_filename):
        self.brightfield_filename = brightfield_filename
        self.dapi_filename = dapi_filename
        brain_image_arr = open_image(self.brightfield_filename)
        self.brain_brightfield = brain_image_arr
        dapi_image_arr = open_image(self.dapi_filename)
        self.brain_dapi = dapi_image_arr

    def segment_brain(self, mask_thresh=0.7, mask_gt_lt="lt", gain=50, gamma=8, hole_size=10000, object_size=20000):
        image_arr = self.brain_brightfield 
        contrast_increased = color.rgb2gray(exposure.adjust_gamma(image_arr, gain=gain, gamma=gamma))
        mask_brain = np.zeros_like(contrast_increased)
        #plt.imshow(contrast_increased)
        #plt.show()
        if mask_gt_lt == "lt":
            mask_brain[contrast_increased < mask_thresh] = 1
        else:
            mask_brain[contrast_increased > mask_thresh] = 1
        #plt.imshow(mask_brain)
        #plt.show()
        mask_brain = remove_small_holes(mask_brain.astype(int), hole_size)
        #plt.imshow(mask_brain)
        #plt.show()
        labelled_brain = measure.label(mask_brain)
        labelled_brain = measure.label(remove_small_objects(labelled_brain, object_size))
        #plt.imshow(labelled_brain)
        #plt.show()
        self.brain = get_largest_connected_component(labelled_brain)
        brain_contour = measure.find_contours(labelled_brain, 0.5)
        contour_image_brain = np.zeros_like(mask_brain)
        for i in range(len(brain_contour)):
            for j in range(len(brain_contour[i])):
                contour_image_brain[int(brain_contour[i][j][0]), int(brain_contour[i][j][1])] = 1
        contour_image_brain = binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(contour_image_brain)))))
        self.brain_contours = contour_image_brain

    def locate_tumour(self, marker_threshold=0.15, gain1=10, gamma1=4, sigma=20, gain2=30, gamma2=4):
        brain_image_arr = self.brain_dapi
        contrast_increased = exposure.adjust_gamma(brain_image_arr, gain=gain1, gamma=gamma1)
        #plt.imshow(contrast_increased)
        #plt.show()
        tumour_gaussian = gaussian(contrast_increased, sigma=sigma)
        #plt.imshow(tumour_gaussian)
        #plt.show()
        # increase contrast and blurring again
        contrast_increased2 = exposure.adjust_gamma(tumour_gaussian, gain=30, gamma=4)
        #plt.imshow(contrast_increased2)
        #plt.show()
        tumour_gs = color.rgb2gray(contrast_increased2)
        print("tumour gs")
        #plt.imshow(tumour_gs)
        #plt.show()
        markers = np.zeros_like(tumour_gs)
        markers[tumour_gs > marker_threshold] = 1
        #plt.imshow(markers)
        #plt.show()
        largest_label = get_largest_connected_component(markers)
        #plt.imshow(largest_label)
        #plt.show()
        return largest_label


    def segment_tumour(self, graphene_threshold=100, marker_threshold=0.15, gain1=10, gamma1=4, sigma=20, gain2=30, gamma2=4):
        bb_arr = np.array(color.rgb2gray(self.brain_brightfield))
        thresholded = np.zeros_like(bb_arr)
        for i in range(len(bb_arr)):
            for j in range(len(bb_arr[i])):
                    if bb_arr[i, j] < graphene_threshold:
                        thresholded[i, j] = bb_arr[i, j]
        tumour = self.locate_tumour(marker_threshold, gain1, gamma1, sigma, gain2, gamma2)
        if len(tumour) != len(thresholded):
            y_dim = np.min([len(tumour), len(thresholded)])
        if len(tumour[0]) != len(thresholded[0]):
            x_dim = np.min([len(tumour[0]), len(thresholded[0])])
        print("thresholded")
        #plt.imshow(thresholded)
        #plt.show()
        tumour_graphene = tumour[:y_dim, :x_dim].astype(int)*thresholded[:y_dim, :x_dim]
        tumour_graphene_bin = np.zeros_like(tumour_graphene)
        tumour_graphene_bin[tumour_graphene>0] = 1
        self.tumour = tumour[:y_dim, :x_dim]
        self.tumour_graphene_bin = tumour_graphene_bin
        tumour_contour = measure.find_contours(tumour, 0.5)
        contour_image = np.zeros((tumour_graphene_bin.shape[0],tumour_graphene_bin.shape[1]))
        for i in range(len(tumour_contour[0])):
            contour_image[int(tumour_contour[0][i][0]), int(tumour_contour[0][i][1])] = 1
        contour_image = binary_dilation(binary_dilation(binary_dilation(contour_image)))
        self.tumour_contours = contour_image

    def calculate_graphene_density(self, window_size):
        window = np.ones((window_size,window_size))
        windowed_average = fftconvolve(self.tumour_graphene_bin, window)
        self.densities = windowed_average
        norm = colors.Normalize(0, vmax=np.max(windowed_average)-60, clip=True)
        m = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        average_colourmapped = m.to_rgba(windowed_average)
        self.average_colourmapped = average_colourmapped
        plt.imshow(average_colourmapped)
        plt.show()
        self.mappable = m

    def make_contour_images(self):
        mask_3d = np.zeros_like(self.average_colourmapped)
        for i in range(len(self.tumour_contours)):
            for j in range(len(self.tumour_contours[i])):
                if self.tumour_contours[i, j] == 1:
                    mask_3d[i,j] = np.array([0, 255,0,1])
                if self.brain_contours[i,j] == 1:
                    mask_3d[i,j] = np.array([255,0,0,1])
        return mask_3d

    def present_results(self, outfile_name):
        mask_3d = self.make_contour_images()
        average_colourmapped = self.average_colourmapped
        for i in range(len(self.tumour)):
            for j in range(len(self.tumour[i])):
                if self.tumour_graphene_bin[i,j] == 1:
                    if average_colourmapped[i,j, 0] == 0 and average_colourmapped[i,j, 1] == 0 and average_colourmapped[i,j, 2] == 0.5 and average_colourmapped[i,j, 3] == 1:
                        pass
                    else:
                        mask_3d[i,j] = average_colourmapped[i,j]
        plt.figure()
        ax = plt.gca()
        im = ax.imshow(mask_3d)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(self.mappable, cax=cax)
        plt.savefig(outfile_name)



image_folder = "/Users/user/Documents/image_analysis/paul/processed_images/"
U87_GO_17_4a = "U87-GO-17-4a/"
U87_GO_26_5a = "U87-GO-26-5a/"
tumour_image_17_4a = "U87-GO-17_4a_x20_all.jpg"
tumour_image_26_5a = "U87-GO-26_overlay.jpg"
go_image = "U87-GO-17_4a_x20_BF"
go_image_26_5a = "U87-GO-26_x4_BF.jpg"

brightfield_image = image_folder + U87_GO_17_4a + brightfield_brain
dapi_image = image_folder + U87_GO_17_4a + whole_brain_image

brightfield_image_26 = image_folder + U87_GO_26_5a + go_image_26_5a
dapi_image_26 = image_folder + U87_GO_26_5a +tumour_image_26_5a
sbi = SegmentBrainImage(brightfield_image_26, dapi_image_26)
sbi.segment_brain()
sbi.segment_tumour()
sbi.calculate_graphene_density(10)
sbi.present_results("test.png")
# default segment_brain values work for U87_GO_17_4a

sbi = SegmentBrainImage(brightfield_image_26, dapi_image_26)
sbi.segment_brain(70, "gt", 7, 8, 100000)
sbi.segment_tumour(graphene_threshold=120, sigma=30)
sbi.calculate_graphene_density(10)
sbi.present_results("26_res")



