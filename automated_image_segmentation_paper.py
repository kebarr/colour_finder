# try this method: https://www.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-24/issue-02/021204/Combining-fluorescence-based-image-segmentation-and-automated-microfluidics-for-ultrafast/10.1117/1.JBO.24.2.021204.full?SSO=1

import skimage.filters

merge4 = "maria/count_cells2/Merge-2.4.jpg"
img_hsv = Image.open(merge4)

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

# blur image
image_rgb = np.array(Image.open(filename))
image_blurred = skimage.filters.gaussian(image_rgb)

subtracted = image_mf.astype(np.int8) - image_blurred.astype(np.int8)
# much better on left


