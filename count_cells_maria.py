import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_erosion, binary_opening, binary_closing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import skimage
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage import measure
from scipy.ndimage import median_filter
from skimage.color import label2rgb

filename = '../maria/green.png'
img = Image.open(filename)
img.load()
image_arr = np.array(img)
#plt.imshow(image_arr)
#plt.show()

filter = lambda x: (50 < x[1])
filter2 = lambda x: (150 < x[1]) | ((3<x[0]<25) & (3 < x[2] < 25) & (40 < x[1] < 100))
green = 0
res = np.zeros((image_arr.shape[0], image_arr.shape[1]))

match_arr = np.zeros((image_arr.shape[0], image_arr.shape[1]))
for i in range(len(image_arr)):
        for j in range(len(image_arr[i])):
                #if filter(image_arr[i][j]):
                #        image_arr[i][j] = np.array([255, 0, 40, 255], dtype=np.uint8)
                #        green+= 1
                if filter2(image_arr[i][j]):
                        image_arr[i][j] = np.array([40, 0, 250, 255], dtype=np.uint8)
                        match_arr[i, j] = 1


green
plt.imshow(image_arr)
plt.show()

erosion = binary_erosion(match_arr)

opening = binary_opening(match_arr) # better than erosion
closing = binary_closing(match_arr) # nope


# might need 2 filters, one for all green (so x[i] > 50) and one just for lime green (x[1] > 150)
# try from here: https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html

filename = '../maria/green.png'
img_gs = Image.open(filename).convert('L')
img_gs.load()
image_arr_gs = np.array(img_gs)

threshold = 80

# make all pixels < threshold black
binarized = 1.0 * (image_arr_gs > threshold)
plt.imshow(binarized)
plt.show()

binarized = ~binarized
binarized[biarized >0] 
from skimage.feature import canny
edges = canny(binarized)
from scipy import ndimage as ndi
# try region based segmentation
markers = np.zeros_like(image_arr_gs)
markers[image_arr_gs < 30] = 1
markers[image_arr_gs > 60] = 2

from skimage.filters import sobel
elevation_map = sobel(image_arr_gs)
from skimage.morphology import watershed
segmentation = watershed(elevation_map, markers) # better than binary opening
from skimage import measure
labels = measure.label(segmentation)
labels.max() # 663

props = measure.regionprops(labels, image_arr_gs)
euler_numbers = [p.euler_number for p in props[1:]]
# euler number of 1= no holes, 
total_cells = 0
for e in euler_numbers:
    if e == 0 or e == 1:
        total_cells += 1
    else:
        # euler number is 1 - number of holes, so number of cells is -euler_number + 1
        total_cells += abs(e) + 1

    
# 668

# first is background, so ignore
seg = segmentation -1
labels2, _ = ndi.label(seg)


skel = skimage.morphology.skeletonize(binarized) # nope

contours = measure.find_contours(image_arr_gs, 100) # try varying fully connected/positive orientation
contours = measure.find_contours(image_arr_gs, 50, fully_connected='high') #positive_orientation
contours = measure.find_contours(image_arr_gs, 50, fully_connected='low', positive_orientation='low')


# low fully connected is good at level 50, high looks similar but finds slightly more contours
# don't want contours in middle of single cells
print(len(contours))
fig, ax = plt.subplots()
ax.imshow(image_arr_gs, cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# contours could work, but there are too many of them, try reducing to 3 or 4 categories to reduce number of 

# euler number!!!!


########## all bits that i actually used in the end:

filename = '../maria/green.png'
img_gs = Image.open(filename).convert('L')
img_gs.load()
image_arr_gs = np.array(img_gs)
markers = np.zeros_like(image_arr_gs)
markers[image_arr_gs < 30] = 1
markers[image_arr_gs > 60] = 2

elevation_map = sobel(image_arr_gs)
segmentation = watershed(elevation_map, markers) # better than binary opening
labels = measure.label(segmentation)
labels.max() # 663

props = measure.regionprops(labels, image_arr_gs)
euler_numbers = [p.euler_number for p in props[1:]]
# euler number of 1= no holes, 
total_cells = 0
for e in euler_numbers:
    if e == 0 or e == 1:
        total_cells += 1
    else:
        # euler number is 1 - number of holes, so number of cells is -euler_number + 1
        total_cells += abs(e) + 1



# now do same for yellow......
filename = '../maria/green_red_yellow.png'
img_gs = Image.open(filename)#.convert('L')
img_gs.load()
image_arr_gs = np.array(img_gs)
markers = np.zeros_like(image_arr_gs)
markers[image_arr_gs < 100] = 1
markers[image_arr_gs > 200] = 2
# see if complicated steps are needed....
lbl = measure.label(markers)
#from skimage.color import label2rgb
lbl_img = label2rgb(lbl, image_arr_gs)
props = measure.regionprops(lbl, image_arr_gs)
euler_numbers = [p.euler_number for p in props[1:]]
# euler number of 1= no holes, 
total_cells_yellow = 0
for e in euler_numbers:
    if e == 0 or e == 1:
        total_cells_yellow += 1
    else:
        # euler number is 1 - number of holes, so number of cells is -euler_number + 1
        total_cells_yellow += abs(e) + 1


def get_segmented_image(filename, marker_lower, marker_upper):
    img_gs = Image.open(filename).convert('L')
    img_gs.load()
    image_arr_gs = np.array(img_gs)
    markers = np.zeros_like(image_arr_gs)
    markers[image_arr_gs < marker_lower] = 1
    markers[image_arr_gs > marker_upper] = 2
    elevation_map = sobel(image_arr_gs)
    segmentation = watershed(elevation_map, markers)


def count_cells(to_count, original_array):
    labels = measure.label(to_count)
    props = measure.regionprops(labels, original_array)
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


# so then we just have
res = count_cells(green_filename, 30, 60) - count_cells(yellow_filename, 100, 200)

### arrgh!! no, cos greyscale is including some green
# so need full colour and filter yellow
filename = '../maria/green_red_yellow.png'
img = Image.open(filename)
img.load()
image_arr = np.array(img)

img_gs = Image.open(filename).convert('L')
img_gs.load()
image_arr_gs = np.array(img_gs)


filter_match = lambda x: ((150<x[0]) & (115 < x[1]) & (35<x[2]<60))
filter_not_match = lambda x:((x[1] < 50))
yellow = 0
res = np.zeros((image_arr.shape[0], image_arr.shape[1]))

match_arr = np.zeros((image_arr.shape[0], image_arr.shape[1]))
for i in range(len(image_arr)):
        for j in range(len(image_arr[i])):
                if filter_match(image_arr[i][j]):
                        image_arr[i][j] = np.array([40, 0, 250, 255], dtype=np.uint8)
                        match_arr[i, j] = 1


plt.imshow(image_arr)
plt.show()
lbl = measure.label(match_arr)
lbl.max()

# then do stuff on match arr
opening = binary_opening(match_arr)
lbl = measure.label(opening)
res = count_cells(lbl, match_arr)
res

# try doing full way, like with green image

markers = np.zeros_like(image_arr_gs)
markers[image_arr_gs < 100] = 1
markers[image_arr_gs > 200] = 2


# DOES NOT WORK FOR YELLOW FINDING
# need to create a 2d array of equivalent markers
elevation_map = sobel(image_arr_gs)
segmentation = watershed(elevation_map, markers) 
labels = measure.label(segmentation)
labels.max()

props = measure.regionprops(labels, image_arr)

### https://stackoverflow.com/questions/30551987/how-do-i-find-and-remove-white-specks-from-an-image-using-scipy-numpy
# best results:::

filtered = median_filter(match_arr, size=3) # orignally done with size = 3
lbl = measure.label(filtered)
lbl.max()
res = count_cells(lbl, match_arr)
res

# need to tweak match array for above as it currently breaks up some of the large bits

def count_yellow(filename):
    img = Image.open(filename)
    img.load()
    image_arr = np.array(img)
    filter_match = lambda x: ((150<x[0]) & (115 < x[1]) & (x[2]<60))
    match_arr = np.zeros((image_arr.shape[0], image_arr.shape[1]))
    for i in range(len(image_arr)):
            for j in range(len(image_arr[i])):
                    if filter_match(image_arr[i][j]):
                        image_arr[i][j] = np.array([40, 0, 250], dtype=np.uint8)
                        match_arr[i, j] = 1
    outfile = filename.split('.jpg')[0] + '_yellow.png'
    filtered = median_filter(match_arr, size=3) # orignally done with size = 3
    lbl = measure.label(filtered)
    lbl_img = label2rgb(lbl, np.array(Image.open(filename).convert('L')))
    plt.imshow(lbl_img)
    plt.show()
    res = count_cells(filtered, match_arr, outfile)
    return res


def get_gs_array(filename):
    img_gs = Image.open(filename).convert('L')
    img_gs.load()
    return np.array(img_gs)

def get_segmented_image(array, marker_lower, marker_upper):
    markers = np.zeros_like(array)
    markers[array < marker_lower] = 1
    markers[array > marker_upper] = 2
    elevation_map = sobel(array)
    ws = watershed(elevation_map, markers)
    return ws


def count_cells(to_count, original_array, outfile, filter=False):
    labels = measure.label(to_count)
    if filter:
        labels = skimage.morphology.remove_small_objects(labels, 10)
    plt.imshow(labels)
    plt.show()
    props = measure.regionprops(labels, original_array)
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



filename_green = '../maria/count_cell_images/' + iba_images[0]
filename_yellow = '../maria/count_cell_images/' + merge_images[0]
cells_yellow = count_yellow(filename_yellow)
green_arr = get_gs_array(filename_green)
green_segmented = get_segmented_image(green_arr, 30, 60)
outfile = filename_green.split('.jpg')[0] + '_green.png'
cells_green = count_cells(green_segmented, green_arr, outfile, True)
final = 100*cells_yellow/cells_green
print("for file %s, %d yellow cells, %d green cells, %f yellow out of green" % (filename_yellow, cells_yellow, cells_green, final))


# possibly useful later: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1458526/

# do the same for red- paul said they have to be adjacent to a red bit and a green bit
#red is 250, 0, 0ß

filename = '../maria/green_red_yellow.png'
img = Image.open(filename)
img.load()
image_arr = np.array(img)

filter_yellow = lambda x: ((150<x[0]) & (115 < x[1]) & (35<x[2]<60))
filter_red = lambda x: (((200< x[0]) & (x[1] < 90) & (x[2]<60)) | (50< x[0] & x[1] < 10 & x[2] <10))
filter_green = lambda x: (150 < x[1]) | ((0<x[0]<25) & (0 < x[2] < 25) & (30 < x[1] < 100) )

for i in range(len(image_arr)):
        for j in range(len(image_arr[i])):
                if filter_yellow(image_arr[i][j]):
                        image_arr[i][j] = np.array([40, 0, 250, 255], dtype=np.uint8)
                if filter_red(image_arr[i][j]):
                        image_arr[i][j] = np.array([255, 20, 147, 255], dtype=np.uint8)
                if filter_green(image_arr[i][j]):
                        image_arr[i][j] = np.array([127, 255, 212, 255], dtype=np.uint8)


plt.imshow(image_arr)
plt.show()

maria_images = os.listdir("../maria/count_cell_images")
# need to pair up merge (yellow, green, red) images with IBA (green) images
merge_images = sorted([im for im in maria_images if im.startswith('Merge')])
iba_images = sorted([im for im in maria_images if im.startswith('IBA')])
for i in range(len(merge_images)):
    filename_green = '../maria/count_cell_images/' + iba_images[i]
    filename_yellow = '../maria/count_cell_images/' + merge_images[i]
    cells_yellow = count_yellow(filename_yellow)
    green_arr = get_gs_array(filename_green)
    green_segmented = get_segmented_image(green_arr, 30, 60)
    outfile = filename_green.split('.jpg')[0] + '_green.png'
    cells_green = count_cells(green_segmented, green_arr, outfile, True)
    final = 100*cells_yellow/cells_green
    print("for file %s, %d yellow cells, %d green cells, %f yellow out of green" % (filename_yellow, cells_yellow, cells_green, final))




merge12_fname = '../maria/count_cell_images/' +merge_images[0]
img_gs = Image.open(merge12_fname)#.convert('L')
img_gs.load()
image_arr_gs = np.array(img_gs)
image_means = np.zeros_like(image_arr_gs)

from scipy.signal import convolve2d

# try to exclude red bands based on means, if that does work use textures
def convolve_all_colours(im, window):
    """
    Convolves im with window, over all three colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = convolve2d(im[:,:,d], window, mode="same", boundary="symm")
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")
    return im_conv

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15,15));
# https://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/
 
w = 40
window = np.ones((w,w))
window /= np.sum(window)
axs.imshow(convolve_all_colours(image_arr_gs, window))
axs.set_title("Mean Filter: window size: {}".format(w))
axs.set_axis_off()
plt.show()

merge12_fname = '../maria/count_cell_images/' +merge_images[0]
img_gs = Image.open(merge12_fname)#.convert('L')
img_gs.load()
image_arr_gs = np.array(img_gs)
res = convolve_all_colours(image_arr_gs, window)
filter_red_band = lambda x: (((70< x[0] <140) & (5< x[1] < 40) & (x[2]<10)))
matches = np.zeros((image_arr_gs.shape[0], image_arr_gs.shape[1]), dtype=bool)
for i in range(len(image_arr_gs)):
        for j in range(len(image_arr_gs[i])):
                if filter_red_band(res[i][j]):
                        image_arr_gs[i][j] = np.array([40, 0, 250], dtype=np.uint8)
                        matches[i,j] = False
                else:
                    matches[i,j]=True

# this works ok, can remove spots, then use binary dilation, big spot is 144-150
matches = skimage.morphology.remove_small_holes(matches, 1000)
matches_opened = binary_opening(matches)
matches_closed = binary_closing(binary_closing(matches))
# try flood fill to fill the top
matches_final = skimage.morphology.flood_fill(np.array((matches_closed), dtype=int), (8, 452), 00)
matches_final = skimage.morphology.flood_fill(matches_final, (80, 509), 0)

# finally!!!!!
# count_yellow code:

img = Image.open(merge12_fname)
img.load()
image_arr = np.array(img)
filter_match = lambda x: ((150<x[0]) & (115 < x[1]) & (x[2]<60))
match_arr = np.zeros((image_arr.shape[0], image_arr.shape[1]))
for i in range(len(image_arr)):
        for j in range(len(image_arr[i])):
            if matches_final[i,j] != 0:
                if filter_match(image_arr[i][j]):
                    image_arr[i][j] = np.array([40, 0, 250], dtype=np.uint8)
                    match_arr[i, j] = 1

plt.imshow(image_arr)
outfile = filename.split('.jpg')[0] + '_yellow.png'
plt.savefig(outfile)
filtered = median_filter(match_arr, size=3) # orignally done with size = 3
#lbl = measure.label(filtered)
#lbl_img = label2rgb(lbl, np.array(Image.open(filename).convert('L')))
#plt.imshow(lbl_img)
#plt.show()
res = count_cells(filtered, match_arr)
# 105




merge12_fname = '../maria/count_cell_images/' +merge_images[0]
img_gs = Image.open(merge12_fname).convert('L')
img_gs.load()
image_arr_gs = np.array(img_gs)

iba12_fname = '../maria/count_cell_images/' + iba_images[0]
# used for green
def get_gs_array(filename):
    img_gs = Image.open(filename).convert('L')
    img_gs.load()
    return np.array(img_gs)

green_gs= get_gs_array(iba12_fname)
def get_segmented_image_new(array, marker_lower, marker_upper):
    markers = np.zeros_like(array)
    markers[array < marker_lower] = 1
    markers[array > marker_upper] = 2
    for i in range(len(markers)):
        for j in range(len(markers[0])):
            if matches_final[i][j] == 0:
                markers[i][j] = 1
    plt.imshow(markers)
    plt.savefig(iba12_fname.split('.jpg')[0] + 'final.png')
    elevation_map = sobel(array)
    return watershed(elevation_map, markers)

green = get_segmented_image_new(green_gs, 30, 60)

cells_green = count_cells(green, green_gs, iba12_fname.split('.jpg')[0] + 'final.png', True)
final = 100*res/cells_green




























# lower band section, x: 400-450, y 430-450, upper, 430-490, 0-30
# accept sections: 310, 151 to 352, 181, and 82, 215 - 110, 245
# need to do more to see if there is real sparation
band_locations = [(400, 430), (430, 0), (460, 30), (300, 480), (450, 380)]
normal_locations = [(300, 150), (80, 215), (260, 200), (270, 95), (45, 20), (3, 340)]
band_patches = [image_arr_gs[400:460, 430:460], image_arr_gs[430:490, 0:30]]
normal_patches = [image_arr_gs[300:360, 150:180], image_arr_gs[80:110, 215:245]]

from skimage.feature import greycomatrix, greycoprops


PATCH_SIZE_x = 60
PATCH_SIZE_y = 30
band_patches = []
for loc in band_locations:
    band_patches.append(image_arr_gs[loc[1]:loc[1] + PATCH_SIZE_y,
                             loc[0]:loc[0] + PATCH_SIZE_x])


normal_patches = []
for loc in normal_locations:
    normal_patches.append(image_arr_gs[loc[1]:loc[1] + PATCH_SIZE_y,
                             loc[0]:loc[0] + PATCH_SIZE_x])

xs = []
ys = []
for patch in (band_patches + normal_patches):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])


for (x, y) in band_locations:
    plt.plot(x + PATCH_SIZE_x / 2, y + PATCH_SIZE_y / 2, 'gs')
for (x, y) in normal_locations:
    plt.plot(x + PATCH_SIZE_x / 2, y + PATCH_SIZE_y / 2, 'bs')

plt.plot(xs[:len(band_patches)], ys[:len(band_patches)], 'go',
        label='Grass')
plt.plot(xs[len(normal_patches):], ys[len(normal_patches):], 'bo',
        label='Sky')
#plt.set_xlabel('GLCM Dissimilarity')
#plt.set_ylabel('GLCM Correlation')

plt.show()

# doesn't work either..... mean is actually better

