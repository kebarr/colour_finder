# try this method: https://www.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-24/issue-02/021204/Combining-fluorescence-based-image-segmentation-and-automated-microfluidics-for-ultrafast/10.1117/1.JBO.24.2.021204.full?SSO=1

import skimage.filters

merge4 = "maria/count_cells2/Merge-2.4.jpg"
img_hsv = np.array(Image.open(merge4).convert('HSV'))

filename='maria/count_cell_images/dapi_staining/iba1dapi2.4.jpg'
image = np.array(Image.open(filename).convert('HSV'))
# apply median filter to colour image: https://stackoverflow.com/questions/256…
image_hue = image[:,:,0]
image_sat = image[:,:,1]
image_val = image[:,:,2]

image_hue_mf = skimage.filters.median(image_hue)
image_sat_mf = skimage.filters.median(image_sat)
image_val_mf = skimage.filters.median(image_val)

# then need to knit together and convert to RGB
image_mf = HSVColor(np.dstack((image_hue_mf, image_sat_mf, image_val_mf)))

def median_filter_colour_image(filename):
    image = np.array(Image.open(filename).convert('HSV'))
    image_hue = image[:,:,0]
    image_sat = image[:,:,1]
    image_val = image[:,:,2]
    image_hue_mf = skimage.filters.median(image_hue)
    image_sat_mf = skimage.filters.median(image_sat)
    image_val_mf = skimage.filters.median(image_val)
    # then need to knit together and convert to RGB
    image_mf = HSVColor(np.dstack((image_hue_mf, image_sat_mf, image_val_mf)))
    return image_mf

def median_filter_colour_image_array(image_array):
    image_hue = image_array[:,:,0]
    image_sat = image_array[:,:,1]
    image_val = image_array[:,:,2]
    image_hue_mf = skimage.filters.median(image_hue)
    image_sat_mf = skimage.filters.median(image_sat)
    image_val_mf = skimage.filters.median(image_val)
    # then need to knit together and convert to RGB
    image_mf = HSVColor(np.dstack((image_hue_mf, image_sat_mf, image_val_mf)))
    return image_mf

# blur image
image_rgb = np.array(Image.open(filename))
image_blurred = skimage.filters.gaussian(image_rgb)

subtracted = image_mf.astype(np.int8) - image_blurred.astype(np.int8)
# much better on left

def sharpen_image_array(image_array):
    median_filtered = median_filter_colour_image_array(image_array)
    image_rgb = HSVColor(image_array)
    image_blurred = skimage.filters.gaussian(image_rgb)
    subtracted = median_filtered.astype(np.int8) - image_blurred.astype(np.int8)
    return subtracted


def sharpen_image(filename):
    median_filtered = median_filter_colour_image(filename)
    image_rgb = np.array(Image.open(filename))
    image_blurred = skimage.filters.gaussian(image_rgb)
    subtracted = median_filtered.astype(np.int8) - image_blurred.astype(np.int8)
    return subtracted

merged_sharpened = sharpen_image(merge4)
subtract_dapi_merged =  merged_sharpened - subtracted
add_dapi_merged = merged_sharpened + subtracted

# looks nothing like theirs.... 
# try just merging then sharpening
combined = HSVColor(image+img_hsv)

# try unsharp masking.... 

# increasing amount makes blue bigger
unsharp_masked = skimage.filters.unsharp_mask(dapi, radius=0.5, amount=0.000001)
# doesn't really help

# try on just yellow and see it helps us to count them....

def get_yellow_channel(filename):
    img_hsv = np.array(Image.open(filename).convert('HSV'))
    yellow_channel = np.zeros_like(img_hsv)
    yellow_channel_bin = np.array([[0 for i in range(img_hsv.shape[0])]for j in range(img_hsv.shape[1])])
    for i in range(img_hsv.shape[0]):
        for j in range(img_hsv.shape[1]):
            if img_hsv[i, j, 0] < 59 and img_hsv[i, j, 0] > 40 and img_hsv[i, j, 2] > 50:
                yellow_channel[i, j] = img_hsv[i, j]
                yellow_channel_bin[i,j] = 1
            else:
                yellow_channel[i,j] = np.array([0,0,0])
    return yellow_channel, yellow_channel_bin


yellow_channel, yellow_channel_bin = get_yellow_channel(merge4)
yellow_sharpened = sharpen_image_array(yellow_channel)

yellow_channel_rgb = HSVColor(yellow_channel)

unsharp_masked = skimage.filters.unsharp_mask(yellow_channel_rgb, radius=5, amount=1)


