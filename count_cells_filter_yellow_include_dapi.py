import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_erosion, binary_opening, binary_closing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import measure
import colorsys
from skimage.morphology import remove_small_objects



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

# corresponding dapi:
dapi24 = 'count_cell_images/dapi_standing/iba1dapi2.4.jpg'

# for dapi i need to remove small objects too
dapi_labelled = count_blue('maria/count_cell_images/dapi_staining/iba1dapi2.4.jpg')

# count blue not really working.... could do similar to yellow with blue


