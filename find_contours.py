###########################################################################################
##########################################FIND CONTOURS####################################
###########################################################################################

# Libraries
import numpy as np
import os

import matplotlib.pyplot as plt

import json
import geojson

import openslide

from tissue_batch_processing import image_from_WSI, find_tissue_elements
from mask_to_contours import png2geojson, save_geojson

import time


def read_json(json_path: str):
    """
    Input: Hovernet json path
    Output: Dictionary with nuclei information
    """
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data

def get_wsis_project(project_path):

    data_project_path = os.path.join(project_path, 'data')

    folder_images = [folder_name for folder_name in os.listdir(data_project_path) if os.path.isdir(os.path.join(data_project_path, folder_name))]

    wsi_names = []
    for folder_image in folder_images:

        # WSI name
        json_path = os.path.join(data_project_path, folder_image, 'server.json')
        try:
            json_data = read_json(json_path)
            wsi_name = json_data['metadata']['name']
            wsi_name = wsi_name[:-5]            # remove ".mrxs"

            wsi_names.append(wsi_name)

        except Exception as e:
            print("")

    return wsi_names


def get_region_to_annotate(geojson_path):

    # Read features
    with open(geojson_path) as f:
        gj = geojson.load(f)
    features = gj['features']

    # Take "Region to annotate" coordinates
    for feature in features:

        if "classification" in feature["properties"]:
            type_annotation = feature["properties"]["classification"]["name"]
            if type_annotation == "Region to annotate":

                coords = feature["geometry"]["coordinates"]
                coords = np.array(coords).squeeze()
                coords = coords[:-1]

                return coords

    return None


###########################################################################################
########################################## MAIN ###########################################
###########################################################################################

# Paths
project_path  = r"path_to_qupath_project"
wsi_root      = r'path_to_wsi'
geojsons_path = r"path_to_wsi_geojson"
output_path   = r'output_path'

# Images of the project
wsis_project = get_wsis_project(project_path)

# Process each image
for wsi_name in wsis_project:

    print("WSI:", wsi_name)

    # Read annotation ("Region to annotate")
    geojson_path = os.path.join(geojsons_path, wsi_name+'.geojson')
    # Record start time
    start_time = time.time()
    region_to_annotate_coords = get_region_to_annotate(geojson_path)

    # WSI (low resolution)
    wsi_path = os.path.join(wsi_root, wsi_name +'.mrxs')
    image, W, H, _ = image_from_WSI(wsi_path)
    factor = int(np.round(H / image.shape[0]))
    region_to_annotate_coords_low_resolution = region_to_annotate_coords // factor

    # Tissue mask (low resolution)
    tissue_mask = find_tissue_elements(image, region_to_annotate_coords_low_resolution) * 255

    # Mask to contours
    contours = png2geojson(tissue_mask, factor)
    save_geojson(contours, wsi_name, output_path)
    end_time = time.time()

    # Calculate total time taken
    total_time = end_time - start_time
    print("Total time taken:", total_time, "seconds")
