# Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

import geojson
from geojson import Feature, Polygon

import openslide

from skimage.measure import label


def format_contour(contour, factor):
    """
    Auxiliary function to pass from the cv2.findContours format to
    an array of shape (N,2). Additionally, the first point is added
    to the end to close the contour.
    """
    new_contour = np.reshape(contour * factor, (-1,2)).tolist()
    new_contour.append(new_contour[0])
    return new_contour

def save_geojson(gson, name, path):
    """
    Save geojson to file path + name.
    """
    with open(os.path.join(path, name + '.geojson'), 'w') as f:
        geojson.dump(gson, f)


def create_geojson(contours):
    """
    Input: List of pairs (contour, label).
        Contour is a list of points starting and finishing in the same point.
        label is an integer representing the class of the cell (1: non-tumour, 2: tumour)
    Returns: A list of dictionaries with the geojson format of QuPath
    """
    #label_dict = ["background", "non-tumour", "tumour", "segmented"]
    label_dict = ["background", "tissue"]
    colour_dict = [-9408287, -9408287]
    features = []
    for contour, label in contours:
        assert(label > 0)
        points = Polygon([contour])
        properties = {
                    "object_type": "annotation",
                    "classification": {
                        "name": label_dict[label],
                        "colorRGB": colour_dict[label]
                    },
                    "isLocked": False
                    }
        feat = Feature(geometry=points, properties=properties)
        features.append(feat)
    return features


def png2geojson(png, factor):

    # Label connected components
    png, num_labels = label(png, background=0, return_num=True)

    total_contours = []
    for lab in range(1, num_labels+1):

        # Binary mask of the label
        mask = png.copy()
        mask[mask != lab] = 0
        mask[mask == lab] = 1
        mask = mask.astype(np.uint8)

        # Contours
        contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        contours_tissue = [contours[i] for i in range(len(contours)) if not (hierarchy[0][i][3] >= 0)]
        contours_tissue = filter(lambda x: len(x[0]) >= 3, [(format_contour(c, factor), 1) for c in contours_tissue])
        
        # contours_tissue = [contours[i] for i in range(len(contours)) if not (hierarchy[0][i][3] >= 0)]
        # contours_holes  = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] >= 0]
        # contours_tissue = filter(lambda x: len(x[0]) >= 3, [(format_contour(c, factor), 1) for c in contours_tissue])
        # contours_holes  = filter(lambda x: len(x[0]) >= 3, [(format_contour(c, factor), 0) for c in contours_holes])
        # contours = filter(lambda x: len(x[0]) >= 3, [(format_contour(c, factor), 1) for c in contours])

        # GeoJSON
        # geojson_tissue = create_geojson(contours_tissue)
        # geojson_holes  = create_geojson(contours_holes)
        # geojson = geojson_tissue + geojson_holes
        geojson = create_geojson(contours_tissue)
        total_contours.extend(geojson)

        del mask

    return total_contours
