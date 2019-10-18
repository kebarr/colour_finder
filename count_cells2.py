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
import os
import skimage

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
    plt.savefig(outfile)
    res = count_cells(filtered, match_arr, outfile)
    return res


def get_gs_array(filename):
    img_gs = Image.open(filename).convert('L')
    img_gs.load()
    return np.array(img_gs)

def get_segmented_image(array, marker_lower, marker_upper):
    plt.imshow(array)
    plt.show()
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
    plt.savefig(outfile)
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

maria_images = os.listdir("../maria/count_cell_images")
# need to pair up merge (yellow, green, red) images with IBA (green) images
merge_images = sorted([im for im in maria_images if im.startswith('Merge')])
iba_images = sorted([im for im in maria_images if im.startswith('IBA')])

filename_green = '../maria/count_cell_images/' + iba_images[2]
filename_yellow = '../maria/count_cell_images/' + merge_images[2]
cells_yellow = count_yellow(filename_yellow)
green_arr = get_gs_array(filename_green)
green_segmented = get_segmented_image(green_arr, 30, 45)
outfile = filename_green.split('.jpg')[0] + '_green.png'
cells_green = count_cells(green_segmented, green_arr, outfile, True)
final = 100*cells_yellow/cells_green
print("for file %s, %d yellow cells, %d green cells, %f yellow out of green" % (filename_yellow, cells_yellow, cells_green, final))

# small bits that have a corresponding nucleus (in dapi image) are to be counted
# so get coordinates of each object (unfiltered) and compare to coordinates of dapi stained obects

dapi_images = os.listdir("../maria/count_cell_images/dapi_staining")
dapi_13 = "../maria/count_cell_images/dapi_staining/"+ dapi_images[3]

# blue- anything where r and g are under 5 and blue is over 50?



lbl_blue = count_blue(dapi_13)
# iterate over green labels
# if they are bigger than some size, use (major_axis_length?, filled_area?, equivalent_diameter?)
# if not, find overlapping dapi stain- how?
# can use slices from regionprops of dapi image in original green image
# dapi_img[props[1].slice[0], props[1].slice[1]]
# exclude overlapped green bits from further analysis to avoid double counting

def get_labelled_image_green(array, marker_lower, marker_upper):
    markers = np.zeros_like(array)
    markers[array < marker_lower] = 1
    markers[array > marker_upper] = 2
    elevation_map = sobel(array)
    ws = watershed(elevation_map, markers)
    labels = measure.label(ws)
    return labels


green_arr = get_gs_array(filename_green)
labels = get_labelled_image_green(green_arr, 30, 45)
props = measure.regionprops(labels, green_arr)
# tiny fleck
prop = props[1000]
# try major axis length of 10 as cut off
props_small_regions = [p for p in props if p.major_axis_length < 7]
props_big_regions = [p for p in props if p.major_axis_length >= 7]
# need to visualise these on original image
green_arr = get_gs_array(filename_green)
for p in props_big_regions[1:]:
    green_arr[p.slice[0], p.slice[1]] = 255


plt.imshow(green_arr)
plt.show()
# sweet, now for each of props small regions, see if it overlaps a dapi stained region
# can do this by using slice over the blue label image and checking whether its all equal to the background label colour

def overlaps_dapi_region(prop, dapi_labels):
    slice_in_dapi_labels = dapi_labels[prop.slice[0], prop.slice[1]]
    if np.any(slice_in_dapi_labels != 0): # nonzero value corresponds to label!!
        return set(slice_in_dapi_labels[np.where(slice_in_dapi_labels != 0)])
    return set() 

overlapping_dapi= set()
for prop in props_small_regions:
    res = overlaps_dapi_region(prop, lbl_blue)
    overlapping_dapi = overlapping_dapi.union(res)

len(overlapping_dapi)


def count_blue(filename):
    img = Image.open(filename)
    img.load()
    image_arr = np.array(img)
    #plt.imshow(image_arr)
    #plt.show()
    filter_match = lambda x: ((x[0]<30) & (x[1]< 40) & (80 < x[2]<255) | (x[0]<5) & (x[1]< 5) & (30 < x[2]<255) )
    match_arr = np.zeros((image_arr.shape[0], image_arr.shape[1]))
    for i in range(len(image_arr)):
            for j in range(len(image_arr[i])):
                    if filter_match(image_arr[i][j]):
                        image_arr[i][j] = np.array([250, 250, 0], dtype=np.uint8)
                        match_arr[i, j] = 1
    plt.imshow(image_arr)
    plt.show()
    lbl = measure.label(match_arr)
    return lbl

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


# actually try not splitting, as big ones should have single nucleus for each
def count_green_inc_dapi(filename_green, filename_blue):
    lbl_blue = count_blue(filename_blue)
    plt.imshow(lbl_blue)
    print("%d blue dots found" % (len(lbl_blue)))
    plt.savefig(filename_blue.split("jpg")[0] + '_found.png')
    green_arr = get_gs_array(filename_green)
    labels = get_labelled_image_green(green_arr, 30, 45)
    #plt.imshow(labels)
    #plt.show()
    props = measure.regionprops(labels, green_arr)
    props_small_regions = [p for p in props if p.major_axis_length < 6 and p.major_axis_length>2.5]
    props_big_regions = [p for p in props if p.major_axis_length >= 6]
    print("small: ", len(props_small_regions))
    print("big: ", len(props_big_regions))
    for p in props_big_regions[1:]:
        green_arr[p.slice[0], p.slice[1]] = 255
    #plt.imshow(green_arr)
    #plt.show()
    overlapping_dapi= set()
    for prop in props_small_regions:
        res = overlaps_dapi_region(prop, lbl_blue)
        overlapping_dapi = overlapping_dapi.union(res)
    number_large_cells = count_cells_simple(props_big_regions)
    print("%d large cells found" % number_large_cells)
    print("found %d green cells in %s"% (len(overlapping_dapi)+number_large_cells, filename_green))

# so now need to repeat that for all of them
#dapi_images = sorted(os.listdir("../maria/count_cell_images/dapi_staining"))[1:]
#iba_images = sorted([im for im in maria_images if im.startswith('IBA') and im.endswith('jpg')])

for i in range(len(iba_images)):
    iba_filename = "../maria/count_cell_images/" + iba_images[i]
    dapi_filename = "../maria/count_cell_images/dapi_staining/" + dapi_images[i]
    count_green_inc_dapi(iba_filename, dapi_filename)

# using 50 as fial arg to green segmentation fn gives best results

# 3.2 messes it up cos htere's some very light patches, which are dead cells, exclude these

td = "../maria/count_cell_images/dapi_staining/" +dapi_images[-2]
ti = "../maria/count_cell_images/" + iba_images[-2]
img = Image.open(td)
img.load()
image_arr = np.array(img)
count_green_inc_dapi(ti, td)

td2 = "../maria/count_cell_images/dapi_staining/" +dapi_images[2]
ti2 = "../maria/count_cell_images/" + iba_images[2]
img = Image.open(td2)
img.load()
image_arr = np.array(img)
count_green_inc_dapi(ti2, td2)
### best results so far:(green_arr, 30, 45) and major axis length 6
# found 523 green cells in ../maria/count_cell_images/IBA1-1.2.jpg
# found 341 green cells in ../maria/count_cell_images/IBA1-1.3.jpg
# found 462 green cells in ../maria/count_cell_images/IBA1-2.1.jpg
# found 530 green cells in ../maria/count_cell_images/IBA1-2.2.jpg
# found 463 green cells in ../maria/count_cell_images/IBA1-2.3.jpg
# found 400 green cells in ../maria/count_cell_images/IBA1-3.1.jpg
# found 310 green cells in ../maria/count_cell_images/IBA1-3.2.jpg
# found 344 green cells in ../maria/count_cell_images/IBA1-3.3.jpg




def count_yellow(filename):
    img = Image.open(filename)
    img.load()
    image_arr = np.array(img)
    filter_match = lambda x: ((150<x[0]) & (115 < x[1]) & (x[2]<60)) # | (100<x[0]) & (50 < x[1]) & (x[2]<20))
    match_arr = np.zeros((image_arr.shape[0], image_arr.shape[1]))
    for i in range(len(image_arr)):
            for j in range(len(image_arr[i])):
                    if filter_match(image_arr[i][j]):
                        image_arr[i][j] = np.array([40, 0, 250], dtype=np.uint8)
                        match_arr[i, j] = 1
    #plt.imshow(image_arr)
    #plt.show()
    outfile = filename.split('.jpg')[0] + '_yellow.png'
    #filtered = median_filter(match_arr, size=3) # orignally done with size = 3
    res = count_cells(match_arr, image_arr, outfile, True, 8)
    return res


def count_cells(to_count, original_array, outfile, filter=False, thresh=10):
    labels = measure.label(to_count)
    if filter:
        labels = skimage.morphology.remove_small_objects(labels, thresh)
    plt.imshow(labels)
    plt.savefig(outfile)
    props = measure.regionprops(labels, to_count)
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

# redo yellow counting with different background:
maria_images = os.listdir("../maria/count_cell_images")
merge_images = sorted([im for im in maria_images if im.startswith('Merge') and im.endswith('-1.jpg')])
for i in range(len(merge_images)):
    filename_yellow = '../maria/count_cell_images/' + merge_images[i]
    #print(filename_yellow)
    cells_yellow = count_yellow(filename_yellow)
    print("%d yellow cells in %s" % (cells_yellow, merge_images[i]))


# yellowfilter not detecting enough of 2.3-1
m23 = '../maria/count_cell_images/' + merge_images[2]
img = Image.open(m23)
img.load()
image_arr = np.array(img)
plt.imshow(image_arr)
plt.show()


#found 505 green cells in ../maria/count_cell_images/IBA1-1.2.jpg
#found 325 green cells in ../maria/count_cell_images/IBA1-1.3.jpg
#found 442 green cells in ../maria/count_cell_images/IBA1-2.1.jpg
#found 514 green cells in ../maria/count_cell_images/IBA1-2.2.jpg
#found 451 green cells in ../maria/count_cell_images/IBA1-2.3.jpg
#found 394 green cells in ../maria/count_cell_images/IBA1-3.1.jpg
#found 298 green cells in ../maria/count_cell_images/IBA1-3.2.jpg
#found 323 green cells in ../maria/count_cell_images/IBA1-3.3.jpg

#219 yellow cells in Merge-1.2-1.jpg - makes 43%
#167 yellow cells in Merge-1.3-1.jpg - 51%
#50 yellow cells in Merge-2.3-1.jpg - 11%
#197 yellow cells in Merge-3.1-1.jpg - 50%
#100 yellow cells in Merge-3.2-1.jpg - 33%
#136 yellow cells in Merge-3.3-1.jpg -42%


# iba to count:
iba1 = "maria/count_cells2/IBA1-1.1.jpeg"
iba2 = "maria/count_cells2/IBA1-2.4.jpeg"

# corresponding dapi:
dapi1 = "maria/count_cell_images/dapi_staining/iba1dapi1.1.jpg"
dapi2 = "maria/count_cell_images/dapi_staining/iba1dapi2.4.jpg"
count_green_inc_dapi(iba1, dapi1)
count_green_inc_dapi(iba2, dapi2)
# yellow to count:

merge1 = "maria/count_cells2/Merge-1.1.jpg"
merge2 = "maria/count_cells2/Merge-2.1.jpg"
merge3 = "maria/count_cells2/Merge-2.2.jpg"
merge4 = "maria/count_cells2/Merge-2.4.jpg"
merges = [merge1, merge2, merge3, merge4]

for m in merges:
    cells_yellow = count_yellow(m)
    print("%d yellow cells in %s" % (cells_yellow, m))
  

#165 yellow cells in maria/count_cells2/Merge-1.1.jpg - 264 green, 62%
#71 yellow cells in maria/count_cells2/Merge-2.1.jpg - 442 green, 16%
#50 yellow cells in maria/count_cells2/Merge-2.2.jpg - 514 green, 10%
#45 yellow cells in maria/count_cells2/Merge-2.4.jpg - 449 green, 10%

#found 264 green cells in maria/count_cells2/IBA1-1.1.jpeg
# found 449 green cells in maria/count_cells2/IBA1-2.4.jpeg

# try just analysing the yellow channel
img_hsv = Image.open(merge4).convert('HSV')
image_arr_hsv = np.array(img_hsv)
plt.imshow(image_arr_hsv)
plt.show()

img = Image.open(merge4)
img.load()
image_arr = np.array(img)
plt.imshow(image_arr)
plt.show()

# need to subtract blue,  as RGB for yellow is equal amounts red and green
blue_channel_subtracted = np.array([[[image_arr[i,j,0], image_arr[i,j,1], 0] for i in range(image_arr.shape[1])] for j in range(image_arr.shape[0])])
# don't think this works cos the image looks totally different, and some yellow just disappears

# try hsl

import colorsys

def HSVColor(img):
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Hdat = []
        Sdat = []
        Vdat = [] 
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB',(r,g,b))
    else:
        return None

from matplotlib.colors import hsv_to_rgb

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
plt.imshow(rgb)
plt.show()
# dividing by rd, gn and bl (which are actually hsl) by 255 results in 
# dividing by nothing 0s the gn and bl channels


# yellow in hsv is about 20 to about 70 (very generous, almost red on one side, green on other)

img_hsv = Image.open(merge4).convert('HSV')
image_arr_hsv = np.array(img_hsv)

# yellow_channel = np.array([[[image_arr[i,j,0], image_arr[i,j,0], 0] for i in range(image_arr.shape[1])] for j in range(image_arr.shape[0])])
yellow_channel = np.zeros_like(image_arr_hsv)
yellow_channel_bin = np.array([[0 for i in range(image_arr_hsv.shape[0])]for j in range(image_arr_hsv.shape[1])])
for i in range(image_arr_hsv.shape[0]):
    for j in range(image_arr_hsv.shape[1]):
        if image_arr_hsv[i, j, 0] < 59 and image_arr_hsv[i, j, 0] > 40:
            yellow_channel[i, j] = image_arr_hsv[i, j]
            yellow_channel_bin[i,j] = 1
        else:
            yellow_channel[i,j] = np.array([0,0,0])


plt.imshow(HSVColor(yellow_channel))
plt.show()
# need to convert back to RGB....
# image_arr[i, j, 0] < 70 and image_arr[i, j, 0] > 30 too lax... mostly red and gren

# now need to create a binary array, dilate to fill some gaps round nuclei, 
fig, axs = plt.subplots(1,2)
axs[0].imshow(yellow_channel_bin)
dilated = binary_dilation(yellow_channel_bin)
axs[1].imshow(dilated)
plt.show()
from skimage.morphology import remove_small_objects



# label first, remove small elements, then dilate
labelled = measure.label(yellow_channel_bin)
binary_denoised = remove_small_objects(labelled, 4)
dilated = binary_dilation(binary_denoised)
plt.imshow(dilated)
plt.show()

# use dilated..... then use the dapi staining again to count cells,

# corresponding dapi:
dapi24 = 'count_cell_images/dapi_standing/iba1dapi2.4.jpg'

# for dapi i need to remove small objects too



