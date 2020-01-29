import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image
from skimage import exposure
from skimage.filters import rank_order

def segment_neun(data):
    lowpass = ndimage.gaussian_filter(data, 4)
    labels = data - lowpass
    mask = labels >= 1
    label_values = np.unique(labels)
    labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
    rescaled = exposure.rescale_intensity(labels, out_range=(0, 255))
    markers = np.zeros_like(rescaled)
    markers[rescaled>200] = 1
    return markers