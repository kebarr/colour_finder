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
                print("initialised slide %s" % (filename))
                self.regions_found = 0


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

        def find_collagen(self, image_array):
                # blue is 12, 18, 173 (middle), 66, 116, 173, lightest
                # need to include lilac, 153,74,173
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


filename = '../luis.png'
img = Image.open(filename)
img.load()
image_arr = np.array(img)

#filter = lambda x: (x[0] < 20) & (x[1] < 60)  #& (150 < x[2])
if x[0] > x[2] 
if x[2] > x[0]
filter = lambda x: (x[0] < 100) & (100 < x[2]<255)
blue = 0
res = np.zeros((image_arr.shape[0], image_arr.shape[1]))
for i in range(len(image_arr)):
        for j in range(len(image_arr[1])):
                if filter(image_arr[i][j]):
                        image_arr[i][j] = np.array([255, 238, 0], dtype=np.uint8)
                        blue += 1


plt.imshow(image_arr)
plt.show()
img_arr[566][646]
#  (x[0] < 150) & (100 < x[2]<255) good but slightly too much purple

filename = '../luis2.png'
img = Image.open(filename)
img.load()
image_arr = np.array(img)

#filter = lambda x: (x[0] < 20) & (x[1] < 60)  #& (150 < x[2])

filter = lambda x: (x[0] < 100) & (100 < x[2]<255)
blue = 0
res = np.zeros((image_arr.shape[0], image_arr.shape[1]))
for i in range(len(image_arr)):
        for j in range(len(image_arr[1])):
                if filter(image_arr[i][j]):
                        image_arr[i][j] = np.array([255, 238, 0], dtype=np.uint8)
                        blue+= 1


plt.imshow(image_arr)
plt.show()

luis = os.listdir('Images_from_luis/katie')

for l in luis:
        if l.endswith('.tif'):
                filename = 'Images_from_luis/' + l
                print(filename)
                img = Image.open(filename)
                img.load()
                image_arr = np.array(img)
                #filter = lambda x: (x[0] < 20) & (x[1] < 60)  #& (150 < x[2])
                filter = lambda x: ((x[0] < 100) & (120 < x[2]<255)| (130 < x[0] < 190) & (150 < x[1] <190) & (190 < x[2]<255) | (100 < x[0] < 120) & (140 < x[1] <165) & (190 < x[2]<255)  | (200 < x[0] < 210) & (205 < x[1] <220) & (220 < x[2]<235))
                blue = 0
                count_pixels = 0
                for i in range(len(image_arr)):
                        for j in range(len(image_arr[i])):
                                count_pixels += 1
                                if filter(image_arr[i][j]):
                                        image_arr[i][j] = np.array([255, 238, 0, ], dtype=np.uint8)
                                        blue+= 1
                plt.imshow(image_arr)
                plt.show()
                #plt.imshow(image_arr)
                #plt.savefig(filename.split('.tif')[0]+"blue_highlighted.png")
                print("%d percent pixels are blue" % (blue/count_pixels))
                print("%d blue pixels in %s" % (blue, filename))


# there's really light blue.... 174, 210, 238
# 232, 235, 229 we don't want
# 215 215 216 don't want
#   quite good: filter = lambda x: ((x[0] < 100) & (100 < x[2]<255)| (130 < x[0] < 200) & (150 < x[1] <210) & (190 < x[2]<255))
# 105, 145, 194 do want 
# 123, 125, 162 don't want
# 198, 202, 200 don't want
# but do want 203, 211, 225
# 115, 163, 212 do want

filename = '../Images_from_luis/65 3 month 20x_Box 2-1 F480 4x_23.tif'
img = Image.open(filename)
img.load()
image_arr = np.array(img)
#plt.imshow(image_arr)
#plt.show()

filter = lambda x: ((x[0] < 100) & (120 < x[2]<255)| (130 < x[0] < 190) & (150 < x[1] <190) & (190 < x[2]<255) | (100 < x[0] < 110) & (140 < x[1] <150) & (190 < x[2]<255))
blue = 0
res = np.zeros((image_arr.shape[0], image_arr.shape[1]))
for i in range(len(image_arr)):
        for j in range(len(image_arr[i])):
                if filter(image_arr[i][j]):
                        image_arr[i][j] = np.array([255, 238, 0, ], dtype=np.uint8)
                        blue+= 1

plt.imshow(image_arr)
plt.show()
plt.imshow(image_arr)


# also want 23,17,70
# 37,20,83
# 31,27,91
# 33, 22, 88
# 22, 17, 75
# 28, 23, 70
#36,34,92

for l in luis:
        if l.endswith('.tif'):
                filename = 'Images_from_luis/katie/' + l
                #print(filename)
                img = Image.open(filename)
                img.load()
                image_arr = np.array(img)
                filter = lambda x: ((20 < x[0] < 40) & (15 < x[1] <35) & (65 < x[2]<95) | (x[0] < 100) & (120 < x[2]<255)| (130 < x[0] < 187) & (150 < x[1] <187) & (192 < x[2]<255) | (100 < x[0] < 120) & (140 < x[1] <165) & (190 < x[2]<255)  | (200 < x[0] < 210) & (205 < x[1] <220) & (220 < x[2]<235))
                blue = 0
                res = np.zeros((image_arr.shape[0], image_arr.shape[1]))
                for i in range(len(image_arr)):
                        for j in range(len(image_arr[i])):
                                if filter(image_arr[i][j]):
                                        image_arr[i][j] = np.array([255, 238, 0, ], dtype=np.uint8)
                                        blue+= 1
                plt.imshow(image_arr)
                #plt.show()
                plt.savefig(filename.split('.tif')[0]+"blue_highlighted.png")
                print("%d blue pixels in %s" % (blue, filename))


filename = 'Images_from_luis/katie/61-3 month RE__01.tif'
img = Image.open(filename)
img.load()
image_arr = np.array(img)
filter = lambda x: ((20 < x[0] < 40) & (15 < x[1] <35) & (65 < x[2]<95) | (x[0] < 100) & (120 < x[2]<255)| (130 < x[0] < 187) & (150 < x[1] <187) & (192 < x[2]<255) | (100 < x[0] < 120) & (140 < x[1] <165) & (190 < x[2]<255)  | (200 < x[0] < 210) & (205 < x[1] <220) & (220 < x[2]<235))
blue = 0
res = np.zeros((image_arr.shape[0], image_arr.shape[1]))
for i in range(len(image_arr)):
        for j in range(len(image_arr[i])):
                if filter(image_arr[i][j]):
                        image_arr[i][j] = np.array([255, 238, 0, ], dtype=np.uint8)
                        blue+= 1
plt.imshow(image_arr)

# just test something.....


filter = lambda x: (x[0] ==1 and x[1] == 1) | (x[2] >10 ) | (x[0]==x[1] and x[1] == x[2])
test_array = np.array([[[1,2,3],[1,2,5],[1,2,18],[6,1,1],[6,1,15],[6,1,12],[1,1,1],[2,2,2],[6,1,12],[6,1,1],[2,2,2],[4,5,6]]])
test_array_bool = np.array(list(map(filter, test_array)))

# to improve performance, try making mask:
filter = lambda x: ((20 < x[0] < 40) & (15 < x[1] <35) & (65 < x[2]<95) | (x[0] < 100) & (120 < x[2]<255)| (130 < x[0] < 187) & (150 < x[1] <187) & (192 < x[2]<255) | (100 < x[0] < 120) & (140 < x[1] <165) & (190 < x[2]<255)  | (200 < x[0] < 210) & (205 < x[1] <220) & (220 < x[2]<235))
bitmap = (((image_arr[:,:,0] > 20) & (image_arr[:,:,0] < 40)) & ((image_arr[:,:,1] > 15) & (image_arr[:,:,1] < 35)) & ((image_arr[:,:,2] > 65) & (image_arr[:,:,2] < 95))) | \
        ((image_arr[:,:,0] < 100) & ((image_arr[:,:,1] > 15) & (image_arr[:,:,1] < 35)) & ((image_arr[:,:,2] > 120) & (image_arr[:,:,2] < 225))) | \
        (((image_arr[:,:,0] > 100) & (image_arr[:,:,0] < 120)) & ((image_arr[:,:,1] > 140) & (image_arr[:,:,1] < 165)) & ((image_arr[:,:,2] > 190) & (image_arr[:,:,2] < 255))) |\
        (((image_arr[:,:,0] > 200) & (image_arr[:,:,0] < 210)) & ((image_arr[:,:,1] > 205) & (image_arr[:,:,1] < 220)) & ((image_arr[:,:,2] > 220) & (image_arr[:,:,2] < 235))) 

image_arr[bitmap==True] = [255, 255, 0]

np.sum(np.where(bitmap==False))
# (130 < x[0] < 187) & (150 < x[1] <187) & (192 < x[2]<255)
# (100 < x[0] < 120) & (140 < x[1] <165) & (190 < x[2]<255) 
# (200 < x[0] < 210) & (205 < x[1] <220) & (220 < x[2]<235))

