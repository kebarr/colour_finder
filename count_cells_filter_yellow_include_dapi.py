import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_erosion, binary_opening, binary_closing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import measure
import colorsys
from skimage.morphology import remove_small_objects


merge4 = "maria/count_cells2/Merge-2.4.jpg"
img_hsv = Image.open(merge4).convert('HSV')
image_arr_hsv = np.array(img_hsv)
plt.imshow(image_arr_hsv)
plt.show()

def count_blue(filename):
    img = Image.open(filename)
    img.load()
    image_arr = np.array(img)
    #plt.imshow(image_arr)
    #plt.show()
    filter_match = lambda x: ((x[0]<30) & (x[1]< 40) & (80 < x[2]<255) | (x[0]<5) & (x[1]< 55) & (30 < x[2]<255) )
    match_arr = np.zeros((image_arr.shape[0], image_arr.shape[1]))
    for i in range(len(image_arr)):
            for j in range(len(image_arr[i])):
                    if filter_match(image_arr[i][j]):
                        image_arr[i][j] = np.array([250, 250, 0], dtype=np.uint8)
                        match_arr[i, j] = 1
    plt.imshow(image_arr)
    plt.show()
    lbl = measure.label(match_arr)
    final = remove_small_objects(lbl, 4)
    return final




def HSVColor(img_arr):
        new_arr = np.zeros_like(img_arr)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                hue = img_arr[i, j, 0]
                sat = img_arr[i, j, 1]
                val = img_arr[i, j, 2]
                #print("h, s,v: ", hue, sat, val)
                r, g, b = colorsys.hsv_to_rgb(np.float(hue/255),np.float(sat/255),np.float(val/255))
                #print(r,g,b)
                #print(r*255,g*255,b*255)
                new_arr[i, j, 0] = int(r*255)
                new_arr[i, j, 1] = int(g*255)
                new_arr[i, j, 2] = int(b*255)           
        return new_arr


rgb = HSVColor(np.array(img_hsv))

yellow_channel2 = np.zeros_like(image_arr_hsv)
yellow_channel_bin2 = np.array([[0 for i in range(image_arr_hsv.shape[0])]for j in range(image_arr_hsv.shape[1])])
for i in range(image_arr_hsv.shape[0]):
    for j in range(image_arr_hsv.shape[1]):
        if image_arr_hsv[i, j, 0] < 59 and image_arr_hsv[i, j, 0] > 40 and image_arr_hsv[i, j, 2] > 50:
            yellow_channel2[i, j] = image_arr_hsv[i, j]
            yellow_channel_bin2[i,j] = 1
        else:
            yellow_channel2[i,j] = np.array([0,0,0])


plt.imshow(HSVColor(yellow_channel2))
plt.show()

# corresponding dapi:
dapi24 = 'count_cell_images/dapi_standing/iba1dapi2.4.jpg'

# for dapi i need to remove small objects too
dapi_labelled = count_blue('maria/count_cell_images/dapi_staining/iba1dapi2.4.jpg')

# count blue not really working.... could do similar to yellow with blue

# try this: https://github.com/luispedro/python-image-tutorial/blob/master/Segmenting%20cell%20images%20(fluorescent%20microscopy).ipynb
filename='maria/count_cell_images/dapi_staining/iba1dapi2.4.jpg'
image = np.array(Image.open(filename).convert('L'))

T_otsu = mh.otsu(image)
print(T_otsu)
plt.imshow(image > T_otsu)
# doesn't differentiate between blue and green

img_hsv = Image.open(filename).convert('HSV')
image_arr_hsv = np.array(img_hsv)
plt.imshow(image_arr_hsv)
plt.show()

blue_channel = np.zeros_like(image_arr_hsv)
blue_channel_bin = np.array([[0 for i in range(image_arr_hsv.shape[0])]for j in range(image_arr_hsv.shape[1])])
for i in range(image_arr_hsv.shape[0]):
    for j in range(image_arr_hsv.shape[1]):
        if image_arr_hsv[i, j, 0] < 270 and image_arr_hsv[i, j, 0] > 160:# and image_arr_hsv[i,j,2] > 150:
            blue_channel[i, j] = image_arr_hsv[i, j]
            blue_channel_bin[i,j] = 1
        else:
            blue_channel[i,j] = np.array([0,0,0])

plt.imshow(HSVColor(blue_channel))

blue_channel_greyscale = np.array(Image.fromarray(HSVcolor(blue_channel)).convert("L"))
# now try mahoto stuff....
T_otsu = mh.otsu(blue_channel_greyscale)
print(T_otsu)
plt.imshow(blue_channel_greyscale > T_otsu)
# automatic threshold doesn't work....

T_mean = blue_channel_greyscale.mean()
print(T_mean)
plt.imshow(blue_channel_greyscale > T_mean)
# still not great, try doing with stdev
T_mean_stdev = blue_channel_greyscale.mean() + blue_channel_greyscale.std()/2
print(T_mean_stdev)
plt.imshow(blue_channel_greyscale > T_mean_stdev)

# have similar issue to with Matts, different parts of the image have different properties
# so to the left, things we need to exclude are the same colour as things we need to keep on the right

# just try it all the way through with what we have... see if separating touching cells step helps

blue_channel_greyscalef = mh.gaussian_filter(blue_channel_greyscale, 2.)
T_mean = blue_channel_greyscalef.mean()
bin_image = blue_channel_greyscalef > T_mean
plt.imshow(bin_image)

labeled, nr_objects = mh.label(bin_image)
print(nr_objects)

plt.imshow(labeled)
plt.jet()
# nope

maxima = mh.regmax(mh.stretch(blue_channel_greyscalef))
maxima,_= mh.label(maxima)
# this works for counting, roughly, but we need ovrlap beween dapi stain and  

dist = mh.distance(bin_image)
dist = 255 - mh.stretch(dist)
watershed = mh.cwatershed(dist, maxima)
plt.imshow(watershed)

# try scipy.ndimage distance transform
from scipy.ndimage.morphology import distance_transform_edt
res = distance_transform_edt(blue_channel_greyscale) # better....

# try this directly on the yellow

yellow_channel_greyscale = np.array(Image.fromarray(yellow_channel).convert("L"))


yellow_channel_greyscalef = mh.gaussian_filter(yellow_channel_greyscale, 2.)
T_mean = yellow_channel_greyscalef.mean()
bin_image = yellow_channel_greyscalef > T_mean
plt.imshow(bin_image)

yellow_channel_rgb = HSVColor(yellow_channel)
T_otsu = mh.otsu(yellow_channel_greyscale)
print(T_otsu)
plt.imshow(yellow_channel_greyscale > T_otsu)

# the yellow greyscale actually looks reasonable


def check_sigma(array, sigma):
    arrayf = mh.gaussian_filter(array.astype(float), sigma)
    maxima = mh.regmax(mh.stretch(arrayf))
    maxima = mh.dilate(maxima, np.ones((5,5)))
    plt.imshow(arrayf)
    #plt.imshow(maxima)
    #plt.imshow(mh.as_rgb(np.maximum(255*maxima, arrayf), arrayf, array > T_mean))

res = distance_transform_edt(yellow_channel_greyscale)
dist = 255 - mh.stretch(res)
watershed = mh.cwatershed(dist, maxima)
plt.imshow(watershed)

# make marker array for labelling
markers = np.zeros_like(yellow_channel_greyscalef)
markers[yellow_channel_greyscalef<1] = 1
markers[yellow_channel_greyscalef>30] = 2

# nope
elevation_map = sobel(yellow_channel_greyscalef)
ws = watershed(elevation_map, markers)

from skimage.feature import peak_local_max
distance = distance_transform_edt(yellow_channel_greyscalef)

# insanely slow....90p[]\]
local_maxi = peak_local_max(
    -distance, indices=False, footprint=np.ones((3, 3)))#, labels=yellow_channel_greyscalef)


# also doesn't work.....
markers = ndimage.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

# the larger you make sigma, the more blurred it gets
ycf = ndimage.gaussian_filter(yellow_channel_greyscale2, 1)
bin_image = np.zeros_like(ycf)
bin_image[ycf>12] = 1

# think that will be ok.... 
labelled = measure.label(bin_image)
labels = skimage.morphology.remove_small_objects(labelled, 30)


maxima = mh.regmax(mh.stretch(ycf))
maxima,_= mh.label(maxima)
plt.imshow(maxima)


dist = mh.distance(bin_image)
plt.imshow(dist)

dist = 255 - mh.stretch(dist)
watershed = mh.cwatershed(dist, maxima)
plt.imshow(watershed)
watershed *= bin_image
plt.imshow(watershed)


# gives best results:
yellow_channel_greyscale2 = np.array(Image.fromarray(yellow_channel2).convert("L"))
ycf = ndimage.gaussian_filter(yellow_channel_greyscale2, 1)
bin_image = np.zeros_like(ycf)
bin_image[ycf>12] = 1

# think that will be ok.... 
labelled = measure.label(bin_image)
labels = skimage.morphology.remove_small_objects(labelled, 30)
# gives 763 though.... way too many, try same approach with dapi...


bcf = ndimage.gaussian_filter(blue_channel_greyscale, 1)
bin_image2 = np.zeros_like(bcf)
bin_image2[bcf>10] = 1

# think that will be ok.... 
labelled_b = measure.label(bin_image2)
labels_b = skimage.morphology.remove_small_objects(labelled_b, 30)


# need to take the two contrast approach with this one....
blue_channel_rgb = HSVColor(blue_channel)
# gamma 5 seems ok for left hand side

# gamma below 1 makes the image brighter, but joins up too much stuff
gamma_corrected = exposure.adjust_gamma(blue_channel_rgb, 0.95, 0.95)
logarithmic_corrected = exposure.adjust_log(blue_channel_rgb, 10)
# maybe exact dapi position doesn't matter- if there's dapi thre then there's a nucleus

median_filtered = median_filter_colour_image_array(np.array(Image.fromarray(gamma_corrected).convert("HSV")))

unsharp_masked = skimage.filters.unsharp_mask(gamma_corrected, radius=5, amount=20, multichannel=True)

# now try to select dapi staining from there- so exclude cyan (0,1,1)
# rgb has been scaled to 0, 1, 1
# allow pink (1,0,1) but not purple, (0.509, 0, 1)

# see if grayscale will make it easier.
from skimage.color import rgb2gray

unsharp_masked_gs = rgb2gray(unsharp_masked) # nope.....



blue_channel_rgb = HSVColor(blue_channel)
gamma_corrected = exposure.adjust_gamma(blue_channel_rgb, 0.95, 0.8)
dapi_filter = lambda x: ((x[2] > 0.7))
unsharp_masked = skimage.filters.unsharp_mask(gamma_corrected, radius=5, amount=20, multichannel=True)
match_arr = np.zeros((unsharp_masked.shape[0], unsharp_masked.shape[1]))
for i in range(len(unsharp_masked)):
        for j in range(len(unsharp_masked[i])):
                if dapi_filter(unsharp_masked[i][j]):
                    unsharp_masked[i][j] = np.array([1, 1, 1], dtype=np.uint8)
                    match_arr[i, j] = 1

# now try label, remove small objects, fill holes, convert back to binary, 

# simpler labelling shouldn't need the above
labelled = measure.label(match_arr)

labelling_final = measure.label(remove_small_objects(labelled, 10))

# test function on all other dapi stainings
dapi_dir = 'maria/count_cell_images/dapi_staining/'
dapi_image_files = [filename for filename in os.listdir(dapi_dir) if filename.endswith(".jpg")] 

dapi_labels = []
for filename in dapi_image_files:
    file_path = dapi_dir + filename
    print(file_path)
    dapi = HSVColor(isolate_blue(file_path))
    labeled = label_dapi(dapi)
    print("found %d cells in %s" % (np.max(labeled), file_path))
    #plt.imshow(labeled)
    #plt.show()
    plt.imshow(labeled)
    plt.savefig(dapi_dir + filename.split('.jpg')[0] + 'found.png')
    dapi_labels.append(labeled)


#found 812 cells in maria/count_cell_images/dapi_staining/iba1dapi3.2.jpg
#found 782 cells in maria/count_cell_images/dapi_staining/iba1dapi3.3.jpg
#found 1000 cells in maria/count_cell_images/dapi_staining/iba1dapi1.1.jpg
#found 923 cells in maria/count_cell_images/dapi_staining/iba1dapi1.3.jpg
#found 591 cells in maria/count_cell_images/dapi_staining/iba1dapi3.1.jpg
#found 862 cells in maria/count_cell_images/dapi_staining/iba1dapi1.2.jpg
#found 768 cells in maria/count_cell_images/dapi_staining/iba1dapi2.1.jpg
#found 757 cells in maria/count_cell_images/dapi_staining/iba1dapi2.2.jpg
#found 752 cells in maria/count_cell_images/dapi_staining/iba1dapi2.3.jpg
#found 793 cells in maria/count_cell_images/dapi_staining/iba1dapi2.4.jpg


# now try combining information from blue and yellow channels
merge4 = "maria/count_cells2/Merge-2.4.jpg"
img_hsv = Image.open(merge4).convert('HSV')
image_arr_hsv = np.array(img_hsv)

yellow_channel = np.zeros_like(image_arr_hsv)
yellow_channel_bin = np.array([[0 for i in range(image_arr_hsv.shape[0])]for j in range(image_arr_hsv.shape[1])])
for i in range(image_arr_hsv.shape[0]):
    for j in range(image_arr_hsv.shape[1]):
        if image_arr_hsv[i, j, 0] < 50 and image_arr_hsv[i, j, 0] > 37 and image_arr_hsv[i, j, 2] > 50 and image_arr_hsv[i, j, 1] > 100:
            yellow_channel[i, j] = image_arr_hsv[i, j]
            yellow_channel_bin[i,j] = 1
        else:
            yellow_channel[i,j] = np.array([0,0,0])

yellow_channel_greyscale = np.array(Image.fromarray(yellow_channel).convert("L"))
ycf = ndimage.gaussian_filter(yellow_channel_greyscale, 1)
bin_image = np.zeros_like(ycf)
bin_image[ycf>12] = 1

bin_labelled = measure.label(bin_image)
bin_labelled_final = remove_small_objects(bin_labelled, 15)

# now need to co-localise with dapi
props = measure.regionprops(bin_labelled_final, ycf)

# want to draw the dapi regions where yellow has been found












import skimage.segmentation as seg
image_slic = seg.slic(HSVColor(image_arr_hsv),n_segments=2000)

image_felzenszwalb = seg.felzenszwalb(HSVColor(image_arr_hsv)) 

