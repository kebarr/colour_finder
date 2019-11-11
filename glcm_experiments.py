import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import color

# try glcm for segmentation 
# https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html#sphx-glr-auto-examples-features-detection-plot-glcm-py

def open_image(filename):
    img = Image.open(filename)
    img.load()
    return np.array(img)


image_folder = "/Users/user/Documents/image_analysis/paul/processed_images/"
U87_GO_17_4a = "U87-GO-17-4a/"
U87_GO_26_5a = "U87-GO-26-5a/"
tumour_image_17_4a = "U87-GO-17_4a_x20_all.jpg"
tumour_image_26_5a = "U87-GO-26_overlay.jpg"
go_image = "U87-GO-17_4a_x20_BF.jpg"
go_image_26_5a = "U87-GO-26_x4_BF.jpg"

img_26_5a = open_image(image_folder+U87_GO_26_5a+go_image_26_5a)
img_17_4a = open_image(image_folder+U87_GO_17_4a+go_image)

brain_locations = [(381,422), (495, 1207), (1247, 2005), (1267, 1234), (1435, 2830)]
background_locations = [(66, 36), (46, 334), (1508, 26), (1736, 3092), (180, 2985)]
PATCH_SIZE = 40

brain_patches = []
for loc in brain_locations:
    brain_patches.append(img_26_5a[loc[0]:loc[0]+PATCH_SIZE, loc[1]:loc[1]+PATCH_SIZE])

background_patches = []
for loc in background_locations:
    background_patches.append(img_26_5a[loc[0]:loc[0]+PATCH_SIZE, loc[1]:loc[1]+PATCH_SIZE])



# compute some GLCM properties each patch
xs = []
ys = []
for patch in (brain_patches + background_patches):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(img_26_5a, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in brain_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in background_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(brain_patches)], ys[:len(brain_patches)], 'go',
        label='Brain')
ax.plot(xs[len(background_patches):], ys[len(background_patches):], 'bo',
        label='Background')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(brain_patches):
    ax = fig.add_subplot(3, len(brain_patches), len(brain_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('Brain %d' % (i + 1))

for i, patch in enumerate(background_patches):
    ax = fig.add_subplot(3, len(background_patches), len(background_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('Background %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()


# dissimilarity much lower for background than for brain
# try just looping over and calculating dissimilarity for each


# does segmentation really well, but far too slow

dissimilarities = []
dissimilarity_image = color.gray2rgb(np.zeros_like(img_26_5a))

for i in range(len(img_26_5a)-PATCH_SIZE):
    for j in range(len(img_26_5a[i]) - PATCH_SIZE):
        patch = img_26_5a[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
        dissimilarities.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        res = greycoprops(glcm, 'dissimilarity')[0, 0]
        # naive way to scale colour with dissimilarity
        if res < 4:
            dissimilarity_image[i,j] = np.array([200/(4-res), 0, 0])
        else:
            dissimilarity_image[i,j] = np.array([0,200/(20-res), 0])


# basic histogram doesn't work cos all values are normally distributed

# going over every pixel is incredibly slow, try every 5
dissimilarities = []
#dissimilarity_image = color.gray2rgb(np.zeros((int(img_26_5a.shape[0]/5)+1, int(img_26_5a.shape[1]/5)+1)))
dissimilarity_image = color.gray2rgb(np.zeros((img_26_5a.shape[0], img_26_5a.shape[1])))

for i in range(int((len(img_26_5a)-PATCH_SIZE)/5)-1):
    for j in range(int((len(img_26_5a[i]) - PATCH_SIZE)/5)-1):
        i1 = i*5
        j1 = j*5
        patch = img_26_5a[i1:i1+PATCH_SIZE, j1:j1+PATCH_SIZE]
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
        dissimilarities.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        res = greycoprops(glcm, 'dissimilarity')[0, 0]
        # naive way to scale colour with dissimilarity
        if res < 4:
            dissimilarity_image[i, j] = np.array([20*(4-res), 0, 0])
        elif res < 35:
            dissimilarity_image[i, j] = np.array([0,10*(20-res), 0])
        else:
            dissimilarity_image[i, j] = np.array([0,0, 5*(40-res)])
    if i%100 == 0:
        print(i, i1, j, j1)



# there are faster ways of estimating the variance in a patch....
# try just with standard dev

deviations = []
deviations_image = color.gray2rgb(np.zeros_like(img_26_5a))

for i in range(len(img_26_5a)-PATCH_SIZE):
    for j in range(len(img_26_5a[i]) - PATCH_SIZE):
        patch = img_26_5a[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
        stdev = np.std(patch)
        deviations.append(stdev)
        # naive way to scale colour with dissimilarity
        if stdev < 3:
            deviations_image[i,j] = np.array([int(200/(3-stdev)), 0, 0])
        else:
            deviations_image[i,j] = np.array([0,int(200/20-stdev), 0])


from skimage.filters import gabor
from skimage.morphology import binary_dilation, remove_small_holes, remove_small_objects

# much quicker

filt_real, filt_imag = gabor(img_26_5a, frequency=0.7)
labelled = binary_dilation(measure.label(filt_imag))
filled = remove_small_holes(labelled, 500000)
final = remove_small_objects(filled, 1000)
#https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html#sphx-glr-auto-examples-features-detection-plot-gabor-py
