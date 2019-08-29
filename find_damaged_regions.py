import openslide
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_erosion
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import fftpack


# file format is mrxs, which is a virtual slide, first need to open it and get the data

#cd /Users/user/Documents/image_analysis/maria

filename = "ImagesHEforKatie/R6-S1-2019-08-12T10-04-01.mrxs"

class DamagedRegionFinder(object):
        def __init__(self, filename):
                self.filename = filename
                self.prepare()
                print("initialised slide")
                self.regions_found = 0

        def prepare(self):
                os = openslide.OpenSlide(self.filename)
                images = os.associated_images
                image = images['macro'] # full image without cross hatching on thumbnail
                image = image.convert("RGB") # need full rgb as lose detail in greyscale
                self.im_array = np.array(image)

        def get_image_for_analysis(self, region_y, region_x, len_y=24, len_x=10.5):
                # image is 24cm, 1st layer (down), is 2-5sm, 6.7-9.9cm, 12-15, 17.5-20.1, 21- 24, 
                # across: 10.5cm, 1st col, 1.5 -45, second 4.8-8
                # image size is (1458, 3308)
                print(self.im_array.shape[0], len_y, region_y, region_x)
                scale_factor_y = self.im_array.shape[0]/len_y
                scale_factor_x = self.im_array.shape[1]/len_x
                test_sample = self.im_array[int(scale_factor_y*region_y[0]):int(scale_factor_y*region_y[1])-1, int(scale_factor_x*region_x[0]):int(scale_factor_x*region_x[1])-1]
                test_image = Image.fromarray(test_sample)
                blurred = self.get_blurred_image(test_image)
                return blurred

        def get_blurred_image(self, image):
                kernel = np.array([np.array([1/81 for i in range(9)]) for j in range(9)])
                offset = len(kernel) // 2
                kernel_ft = fftpack.fft2(kernel, shape=np.array(image).shape[:2], axes=(0, 1))
                # convolve
                img_ft = fftpack.fft2(np.array(image), axes=(0, 1))
                # the 'newaxis' is to match to color direction
                img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
                img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real
                # clip values to range
                img2 = np.clip(img2, 0, 255)
                output_image = img2.astype(int)
                return output_image


        def remove_edges(self, results_array):
                # not ideal cos doesn't remove only edges but think extra bits it leaves in make up for bits it shouldn't erode
                erosion_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
                return binary_erosion(results_array, erosion_kernel)

        def find_damaged_region(self, image_array):
                filter = lambda x: (121 < x[0] < 207) & (100 < x[1] < 130) & (145 < x[2] < 160)
                res = np.zeros((image_array.shape[0], image_array.shape[1]))
                for i in range(len(image_array)):
                        for j in range(len(image_array[1])):
                                res[i][j] = filter(image_array[i][j])
                return self.remove_edges(res)


        # so now just need total nonzero pixels, and total pixels found
        def get_sample(self, image):
                filter_sample = lambda x: (180 > x[1]) & (180 > x[2]) # this misses some of the detail but should be ok
                sample = np.zeros((image.shape[0], image.shape[1]))
                for i in range(len(image)):
                        for j in range(len(image[1])):
                                sample[i][j] = filter_sample(image[i][j])
                return sample

        def run(self, region_y, region_x):
                # break this up so don't reload each time
                image = self.get_image_for_analysis(region_y, region_x)
                print("opened image")
                image_arr = np.array(image)
                sample_region = self.get_sample(image_arr)
                damaged_region = self.find_damaged_region(image_arr)
                damaged_area = np.sum(damaged_region)/np.sum(sample_region)
                plt.imshow(damaged_region)
                if '/' in self.filename:
                        filename_base = self.filename.split("/")[1].split(".")[0]
                else:
                        filename_base = self.filename.split(".")[0]
                plt.savefig("damaged_"+ str(self.regions_found) + filename_base)
                plt.imshow(sample_region)
                plt.savefig("sample_" + str(self.regions_found) +  filename_base)
                self.regions_found += 1
                print("Area of: %s region %d : %d, %d:%d is %f" % (self.filename, region_y[0], region_y[1], region_x[0], region_x[1], damaged_area))


files = os.listdir("mrxs_fileshistologyheart_tissue")
# need to define regions, not necessarily exactly same for each so make as big as possible
# image is 24cm, 1st layer (down), is 2-5sm, 6.7-9.9cm, 12-15, 17.5-20.1, 21- 24, 
# across: 10.5cm, 1st col, 1.5 -45, second 4.8-8

# so do 1-4.6 and 4.7-9 for columns
# rows: 1.5 - 5.9, 6-10.9,11-16.3,16.4-20.5,20.6-24

regions = [[(1, 4.2), (1.5, 5.9)], [(1, 4.2), (6,10.9)], [(1, 4.2), (11, 16.3)], [(1, 4.2), (16.4,20.5)], [(1, 4.2), (20.6, 24)], [(4.3, 9), (1.5, 5.9)], [(4.3, 9), (6,10.9)], [(4.3, 9), (11, 16.3)], [(4.3, 9), (16.4,20.5)], [(4.3, 9), (20.6, 24)]]
drf = DamagedRegionFinder("mrxs_fileshistologyheart_tissue/" + files[0]) # for this one i get unsupported image format error
drf = DamagedRegionFinder('ImagesHEforKatie/R6-S1-2019-08-12T10-04-01.mrxs')
for region in regions:
        drf.run(region[1], region[0])

drf.run((11, 16.3), (1, 4.2))
# for first, column delimitation is wrong, need to increase column 2 width
drfs = []
for file in files:
        print(file)
        try:
                drfs.append(DamagedRegionFinder("mrxs_fileshistologyheart_tissue/" + file))
        except:
                print("could not read")