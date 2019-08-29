import openslide
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_erosion
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# file format is mrxs, which is a virtual slide, first need to open it and get the data

#cd /Users/user/Documents/image_analysis/maria

filename = "ImagesHEforKatie/R6-S1-2019-08-12T10-04-01.mrxs"

class DamagedRegionFinder(object):
        def __init__(self, filename):
                self.filename = filename

        def get_image_for_analysis(self, region_y, region_x, len_y=24, len_x=10.5):
                os = openslide.OpenSlide(self.filename)
                images = os.associated_images
                image = images['macro'] # full image without cross hatching on thumbnail
                image = image.convert("RGB") # need full rgb as lose detail in greyscale
                im_array = np.array(image)
                # image is 24cm, 1st layer (down), is 2-5sm, 6.7-9.9cm, 12-15, 17.5-20.1, 21- 24, 
                # across: 10.5cm, 1st col, 1.5 -45, second 4.8-8
                # image size is (1458, 3308)
                scale_factor_y = im_array.shape[0]/len_y
                scale_factor_x = im_array.shape[1]/len_x
                test_sample = im_array[int(scale_factor_y*region_y[0]):int(scale_factor_y*region_y[1]), int(scale_factor_x*region_x[0]):int(scale_factor_x*region_x[1])]
                test_image = Image.fromarray(test_sample)
                blurred = np.array(get_blurred_image(test_image))
                return blurred

        def get_blurred_image(self, image, kernel=kernel):
                input_pixels = image.load()
                kernel = np.array([np.array([1/81 for i in range(9)]) for j in range(9)])
                offset = len(kernel) // 2
                output_image = Image.new("RGB", image.size)
                draw = ImageDraw.Draw(output_image)
                # slooow- look at https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.ndimage.filters.convolve.html
                # Compute convolution between intensity and kernels
                for x in range(offset, image.width - offset):
                        for y in range(offset, image.height - offset):
                        acc = [0, 0, 0]
                        for a in range(len(kernel)):
                                for b in range(len(kernel)):
                                xn = x + a - offset
                                yn = y + b - offset
                                pixel = input_pixels[xn, yn]
                                acc[0] += pixel[0] * kernel[a][b]
                                acc[1] += pixel[1] * kernel[a][b]
                                acc[2] += pixel[2] * kernel[a][b]

                        draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
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
                return remove_edges(res)


        # so now just need total nonzero pixels, and total pixels found
        def get_sample(self, image):
                filter_sample = lambda x: (180 > x[1]) & (180 > x[2]) # this misses some of the detail but should be ok
                sample = np.zeros((image.shape[0], image.shape[1]))
                for i in range(len(image)):
                        for j in range(len(image[1])):
                        sample[i][j] = filter_sample(arr[i][j])
                return sample

        def run(self, region_y, region_x):
                image = get_image_for_analysis(self.filename, (6.7, 9.9), (1.5, 4.5))
                print("opened image")
                image_arr = np.array(image)
                sample_region = get_sample(image_arr)
                damaged_region = find_damaged_region(image_arr)
                damaged_area = np.sum(damaged_region)/np.sum(sample_region)
                plt.imshow(damaged_region)
                filename_base = self.filename.split(".")[0]
                plt.savefig("damaged_"+ filename_base)
                plt.imshow(sample_region)
                plt.savefig("sample_" + filename_base)

                