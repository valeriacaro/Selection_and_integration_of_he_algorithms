# Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

import openslide

from scipy import ndimage as nd
from scipy.spatial import distance

from skimage import morphology
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries

from pathlib import Path
from PIL import Image


def image_from_WSI(wsi_path, outputpath=None, level=6):
    
    # Read the whole image in the corresponding level
    wsi = openslide.OpenSlide(wsi_path)
    mpp = np.round(float(wsi.properties['openslide.mpp-x']), 2)
    if mpp == 0.12:
        level = level + 1

    # Coordinates and size of the openslide
    x0, y0 = int(wsi.properties['openslide.bounds-x']), int(wsi.properties['openslide.bounds-y'])
    h = int(int(wsi.properties['openslide.bounds-height'])/(2**level))
    w = int(int(wsi.properties['openslide.bounds-width'])/(2**level))

    # Read image
    image = np.array(wsi.read_region((x0, y0), level, (w, h)))  

    # Mask no elements (alpha channel 0) as background (white)
    image[image[:,:,3] == 0] = [255, 255, 255, 255]
    image = image[:,:,:3]
    
    # Dimensions high resolution
    # W, H = wsi.dimensions
    W = int(wsi.properties['openslide.bounds-width'])
    H = int(wsi.properties['openslide.bounds-height'])

    return image, W, H, mpp


###########################################################################################
########################################## OSCAR ##########################################
###########################################################################################

def postprocessing(mask, size=200):
    """
    Postprocess the mask to obtain an more compact image, without holes and small
    artifacts.
    Args:
      - mask (np.array): Binary image.
      - size (int): size of the objects to remove.
    Returns:
      - postprocessed_mask (np.array): Binary image (mask) without holes and small
                                       objects.
    """
    
    mask = mask.astype(bool)
    
    # Remove small objects (white and black)
    postprocessed_mask = morphology.remove_small_objects(mask, size)
    postprocessed_mask = morphology.remove_small_objects(np.invert(postprocessed_mask), size)
    
    # Hole filling
    postprocessed_mask = morphology.remove_small_holes(postprocessed_mask, size)
    
    postprocessed_mask = np.uint8(postprocessed_mask)*255
    
    return postprocessed_mask


def Thresholding_Background_Segmentation(image, th=0.05, alpha_channel=True, size=200):
    """
    Foreground-background segmentation of an image using thresholding on the tissue
    region.
    Args:
      - image (np.array): Image to be segmented.
      - th (float): Threshold.
      - alpha_channel (bool): If the input image has an alpha channel (RGBA) or not
        (RGB).
    Returns:
      - postprocessed_mask (np.array): Binary mask postprocessed.
    """

    # Special background zone (alpha = 0)
    if alpha_channel:
        xb, yb = np.where(image[:,:,3]==0)
        xt, yt = np.where(image[:,:,3]!=0)

    else:
        xt, yt = np.where(image[:,:,0] > -1)   # all

    # Tissue region
    im = np.float32(image[xt,yt,0:3])

    # Normalized color distance
    white = [255, 255, 255]; black = [0, 0, 0]
    dist = distance.cdist(im, [white])
    max_dist = distance.euclidean(white, black)
    dist = dist/max_dist

    # Background pixels
    ind, _ = np.where(dist < th)
    x_background, y_background = xt[ind], yt[ind]

    # Binary mask (with the segmentation)
    mask = np.zeros(image.shape[0:2])
    mask[x_background, y_background] = 255
    if alpha_channel:
        mask[xb,yb] = 255

    # Post-processing
    postprocessed_mask = postprocessing(np.uint8(mask), size=size)

    return postprocessed_mask


###########################################################################################
####################################### DIGIPATICS ########################################
###########################################################################################

def gray_binarization(image, th=0.95, closing_size=8, median_size=12):
                    
    # Gray image
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255

    # Pre-processing
    img_gray = 1-nd.grey_closing(1-img_gray, closing_size)
    img_gray = nd.median_filter(img_gray, median_size)

    # Binarize
    binary = (img_gray > th)

    return binary


def saturation_binarization(image, th=0.015, closing_size=6, gaussian_sigma=4):
   
    # Saturation
    image_hsv = rgb2hsv(image)
    s = image_hsv[:,:,1]

    # Pre-processing
    s = nd.grey_closing(s, size=closing_size)
    s = nd.gaussian_filter(s, sigma=gaussian_sigma)

    # Binarize
    binary = (s < th)
    
    return binary


def background_others(image):

    # Binary masks
    gray_binary = gray_binarization(image)
    s_binary = saturation_binarization(image)

    # Combine
    s_binary = np.maximum(s_binary, gray_binary)
    background_others_mask = morphology.reconstruction(s_binary, gray_binary, method='erosion')

    return 1 - background_others_mask


def black_objects(image, th=0.3, mean_size=8, erosion_size=12):
   
    # Value
    image_hsv = rgb2hsv(image)
    v = image_hsv[:,:,2]
    
    # Mean filter
    mean_v = cv2.blur(v, (mean_size, mean_size))
    
    # Binarize
    binary = (v > th)
    mean_binary = (mean_v > th)
    
    # Reconstruction
    mean_binary = np.maximum(mean_binary, binary)
    black_objects_mask = morphology.reconstruction(mean_binary, binary, method='erosion')

    # Post-processing
    black_objects_mask = nd.binary_erosion(black_objects_mask, np.ones((erosion_size,erosion_size)))

    return 1 - black_objects_mask


def exclude_objects(mask, min_area=50, max_aspect_ratio=15):
    
    # Labels
    label_mask = label(mask, background=0)
    props = regionprops(label_mask)
    
    exclude_labels = []
    for region in props:
        
        # Background
        if region.label == 0:
            continue
        exclude = False
        
        # Small object
        if region.area < min_area:
            exclude = True
        
        # Proportion
        # major = region.major_axis_length
        # minor = max(region.minor_axis_length, 1)    # if minor=0 --> minor=1
        # if major/minor > max_aspect_ratio:
        #     exclude = True
        
        # Remove this object (label)
        if exclude:
            exclude_labels.append(region.label)

    # Remove the excluded labels
    mask[np.isin(label_mask, exclude_labels)] = 0
            
    return np.uint8(mask)

def pixels_mask_size(W, H, mpp=0.24):

    # Tile size and overlap at the maximum resolution level
    if mpp == 0.24:
        tile_size, overlap = 1024, 100
    else:
        tile_size, overlap = 2048, 200

    # Complete tiles
    tilesxW = math.floor(W/(tile_size-overlap))
    tilesxH = math.floor(H/(tile_size-overlap))

    # Remainder pixels
    remW = W-max(0,tilesxW)*(tile_size-overlap)
    remH = H-max(0,tilesxH)*(tile_size-overlap)
    
    # Add a tile for remainder pixels (if needed)
    if remW > 0:
        tilesxW += 1
    if remH > 0:
        tilesxH += 1

    return tilesxW, tilesxH


def find_tissue_elements(image, coords_region=None):

    # Mask pixels outside region of interest
    if coords_region is not None:
        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, pts=np.int32([coords_region]), color=(255,255,255))
        mask = mask // 255

        image = image * mask

    # Binary mask
    objects_mask = background_others(image)
    black_objects_mask = black_objects(image)
    final_objects = np.maximum(objects_mask - black_objects_mask, 0)       # crosses bigger in black_objects_mask
    final_objects_mask = exclude_objects(final_objects)
    
    return final_objects_mask

