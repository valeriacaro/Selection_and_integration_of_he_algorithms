"""
Breast tumor detection in Hematoxylin & Eosin stained biopsy images
---------------------------------------------------------------------
From the model obtained, make inference with the patches of the ICS images
to get the predictions in them.
"""

#####################################################################################
##################################### LIBRARIES #####################################
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from dataset import *
from epochs import TrainEpoch, ValidEpoch
from sklearn.metrics import precision_score, recall_score, f1_score

################################################################################
################################ CONFIGURATION #################################
################################################################################

# Normalization
mean_training = [0.71972791, 0.54526797, 0.68434116]
std_training  = [0.1973122 , 0.23205373, 0.17320338]

# Encoder network
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'

# Device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda'

# Classes
CLASSES = ['background', 'other', 'tumor', 'insitu']

# Visualization
colors = np.array([[0.        , 0.        , 0.        ],
                   [0.70196078, 1.        , 0.51764706],
                   [1.        , 0.29411765, 0.33725490],
                   [0.51764706, 0.82352941, 1.        ]])
cmap = LinearSegmentedColormap.from_list('Personalized', colors)
patches = [mpatches.Patch(color=colors[i],label=CLASSES[i]) for i in range(len(CLASSES))]

################################################################################
################################## BEST MODEL ##################################
################################################################################

print('Load model')
path_models = "path_to_models"
model_name = 'best_model_name'
path_best_model = os.path.join(path_models, model_name)
best_model = torch.load(path_best_model)

################################################################################
#################################### DATASET ###################################
################################################################################

print('Load dataset')

# Train/validation dataset
#DATA_DIR = "/mnt/work/datasets/ICS/MAMA/TILES/HE/Vall_Hebron/Single resolution 2022/10x/val/"
DATA_DIR = "directory_to_test_data"
images_dir = os.path.join(DATA_DIR, 'Images')
masks_dir  = os.path.join(DATA_DIR, 'Masks')
train_files, valid_files = split_dataset(images_dir)

# Preprocessing
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing_fn.keywords['mean'] = mean_training
preprocessing_fn.keywords['std']  = std_training

# Validation datasets (visualization and process)
valid_dataset_vis = Dataset(images_dir, masks_dir,
                            classes=CLASSES,
                            )

valid_dataset = Dataset(images_dir, masks_dir,
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES,
                        )

################################################################################
################################# PREDICTIONS ##################################
################################################################################

def predict_segmentation(model, dataset, dataset_vis, save_path):
    """
    Predict the segmentation of the images (patches) of the dataset.
    Args:
    - model: model used to predict the segmentations.
    - dataset: dataset with the images to be segmented.
    - dataset_vis: dataset with the original images.
    - save_path: path of the directory to save the masks of the segmentation.
    """

    # Save images path
    save_path = os.path.join(save_path, 'Predicted_masks/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Images to predict
    random.seed(1234)
    indices = random.sample(range(0, len(dataset)), 18)

    precisions = {0: [], 1: [], 2: [], 3: []}
    recalls = {0: [], 1: [], 2: [], 3: []}
    f1_scores = {0: [], 1: [], 2: [], 3: []}

    for i in indices:

        print('    Image', i)

        image, mask = dataset[i]
        image_vis = dataset_vis[i][0].astype('uint8')

        # Predicted mask (segmentation)
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model(x_tensor)
        act = nn.Softmax(dim=1)
        pr_mask = act(pr_mask)
        pr_mask = np.argmax(pr_mask.cpu().detach().numpy(), 1).squeeze()

        # Compute test metrics
        mask_flat = mask.flatten()
        pr_mask_flat = pr_mask.flatten()

        for cls in [0, 1, 2, 3]:
            precision = precision_score(mask_flat, pr_mask_flat, labels=[cls], average='macro', zero_division=0)
            recall = recall_score(mask_flat, pr_mask_flat, labels=[cls], average='macro', zero_division=0)
            f1 = f1_score(mask_flat, pr_mask_flat, labels=[cls], average='macro', zero_division=0)

            precisions[cls].append(precision)
            recalls[cls].append(recall)
            f1_scores[cls].append(f1)

        # Save the predicted mask
        plt.figure(figsize=(40,10))
        plt.subplot(1,3,1); plt.imshow(image_vis); plt.title('Image'); plt.xticks([]); plt.yticks([])
        plt.subplot(1,3,2); plt.imshow(mask, cmap='gray', vmin=0, vmax=len(CLASSES)); plt.title('GT'); plt.xticks([]); plt.yticks([])
        plt.subplot(1,3,3); plt.imshow(pr_mask, cmap='gray', vmin=0, vmax=len(CLASSES)); plt.title('Prediction'); plt.xticks([]); plt.yticks([])
        plt.savefig(save_path+'image_'+str(i)+'.png')
        # plt.show()

    for cls in [0, 1, 2, 3]:
        print(f"Class {cls} Precision: {np.mean(precisions[cls]):.4f}")
        print(f"Class {cls} Recall: {np.mean(recalls[cls]):.4f}")
        print(f"Class {cls} F1-score: {np.mean(f1_scores[cls]):.4f}")


def predict_segmentation_colors(model, dataset, dataset_vis, save_path):
    """
    Predict the segmentation of the images (patches) of the dataset.
    Args:
    - model: model used to predict the segmentations.
    - dataset: dataset with the images to be segmented.
    - dataset_vis: dataset with the original images.
    - save_path: path of the directory to save the masks of the segmentation.
    """

    # Save images path
    save_path = os.path.join(save_path, 'Predicted_masks/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Images to predict
    random.seed(1234)
    indices = random.sample(range(0, len(dataset)), 18)
    for i in indices:

        print('    Image', i)
        image, mask = dataset[i]
        image_vis = dataset_vis[i][0].astype('uint8')

        # Predicted mask (segmentation)
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model(x_tensor)
        act = nn.Softmax(dim=1)
        pr_mask = act(pr_mask)
        pr_mask = np.argmax(pr_mask.cpu().detach().numpy(), 1).squeeze()
        
        # Save the predicted mask
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1); plt.imshow(image_vis); plt.title('Image'); plt.xticks([]); plt.yticks([])
        plt.subplot(1,3,2); plt.imshow(mask, cmap=cmap, vmin=0, vmax=len(CLASSES)-1, interpolation=None); plt.title('GT'); plt.xticks([]); plt.yticks([])
        plt.subplot(1,3,3); plt.imshow(pr_mask, cmap=cmap, vmin=0, vmax=len(CLASSES)-1, interpolation=None); plt.title('Prediction'); plt.xticks([]); plt.yticks([])
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, fontsize=12)
        plt.savefig(save_path+'image_'+str(i)+'.png')
        # plt.show()

import matplotlib.colors as mcolors
def predict_segmentation_overlay(model, dataset, dataset_vis, save_path):

    """
    Predict the segmentation of the images (patches) of the dataset.
    Args:
    - model: model used to predict the segmentations.
    - dataset: dataset with the images to be segmented.
    - dataset_vis: dataset with the original images.
    - save_path: path of the directory to save the masks of the segmentation.
    """

    # Save images path
    save_path = os.path.join(save_path, 'Predicted_masks/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    class_colors = {
        0: (0.95, 0.95, 0.95),  # Light gray for class 0
        1: (0.75, 0.75, 0.75),           
        2: (0.6, 0.4, 0.2),  # Light red for class 2
        3: (0.7, 1, 0.7)     # Green for class 1
    }

    # Images to predict
    random.seed(1234)
    indices = random.sample(range(0, len(dataset)), 18)
    for i in indices:

        print('    Image', i)
        image, mask = dataset[i]
        image_vis = dataset_vis[i][0].astype('uint8')

        # Predicted mask
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model(x_tensor)
        act = nn.Softmax(dim=1)
        pr_mask = act(pr_mask)
        pr_mask = np.argmax(pr_mask.cpu().detach().numpy(), 1).squeeze()
        
        # Overlay the ground truth mask on the original image with transparency
        overlay_gt = image_vis.copy()
        overlay_pred = image_vis.copy()
        alpha = 0.50  # Adjust transparency
        for class_idx, color in class_colors.items():
            if class_idx in [2, 3]:  # Only apply for classes 2 and 3
                mask_indices = mask == class_idx
                pred_mask_indices = pr_mask == class_idx
                overlay_color = np.array(color) * 255
                overlay_gt[mask_indices] = (1 - alpha) * overlay_gt[mask_indices] + alpha * overlay_color
                overlay_pred[pred_mask_indices] = (1 - alpha) * overlay_pred[pred_mask_indices] + alpha * overlay_color

        # Save the overlay GT image
        plt.figure(figsize=(10, 5))
        plt.imshow(overlay_gt.astype(np.uint8))
        plt.title('Original Image with Ground Truth Mask')
        plt.axis('off')
        plt.savefig(save_path + 'image_' + str(i) + '_overlay_gt.png')
        plt.close()
        
        # Save the predicted mask
        plt.figure(figsize=(10, 5))
        plt.imshow(overlay_pred.astype(np.uint8))
        plt.title('Original Image with Predicted Mask')
        plt.axis('off')
        plt.savefig(save_path + 'image_' + str(i) + '_overlay_pred.png')
        plt.close()
        

def predict_segmentation_overlay_mini(model, image):
    """
    Predict the segmentation of the given image.
    Args:
    - model: model used to predict the segmentations.
    - image: image to be segmented.
    Returns:
    - predicted_mask: predicted segmentation mask.
    """

    class_colors = {
        0: (0.95, 0.95, 1),   # Light blue for class 0
        1: (0.7, 1, 0.7),   # Light green for class 1
        2: (1, 0.6, 0.6),   # Light red for class 2
        3: (1, 0.8, 0.5)    # Light orange for class 3
    }
    # Predicted mask (segmentation)
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = model(x_tensor)
    act = nn.Softmax(dim=1)
    pr_mask = act(pr_mask)
    pr_mask = np.argmax(pr_mask.cpu().detach().numpy(), 1).squeeze()

    return pr_mask



import os
import re
from PIL import Image

# Function to extract position information from image filename
def extract_position_info(filename):
    # Extract position information from filename
    match = re.search(r'\[d=(\d+),x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]', filename)
    if match:
        x = int(match.group(2))
        y = int(match.group(3))
        w = int(match.group(4))
        h = int(match.group(5))
        return x, y, w, h
    else:
        return None

# Function to reconstruct the whole image from predicted masks
def reconstruct_image(model, valid_dataset, save_path):
    # Dictionary to store predicted masks with their position information
    predicted_masks = {}

    # Predict masks for each image in the dataset
    for i in range(len(valid_dataset)):
        # Read the image from the dataset
        image, _ = valid_dataset[i]
        filename = valid_dataset.images_fps[i]

        # Extract position information from filename
        position_info = extract_position_info(filename)
        if position_info is not None:
            x, y, w, h = position_info

            # Predict the mask for the image
            predicted_mask = predict_segmentation_overlay_mini(model, image)

            # Store the predicted mask with its position information
            predicted_masks[(x, y, w, h)] = predicted_mask

            print(i)

    scale_factor = 0.2  # Adjust as needed

    # Reconstruct the whole image by placing each predicted mask in the correct position
    max_x = max(x + w for x, y, w, h in predicted_masks.keys())
    max_y = max(y + h for x, y, w, h in predicted_masks.keys())

    # Scale down the dimensions
    scaled_max_x = int(max_x * scale_factor)
    scaled_max_y = int(max_y * scale_factor)

    # Create a new figure
    plt.figure(figsize=(scaled_max_x / 100, scaled_max_y / 100))  # Adjust figure size based on scaled image dimensions

    for (x, y, w, h), mask in predicted_masks.items():
        mask = mask.astype(np.uint8)

        # Resize the mask to match scaled dimensions
        resized_mask = np.array(Image.fromarray(mask).resize((scaled_max_x, scaled_max_y), Image.NEAREST))

        # Place the predicted mask in the correct position on the full image
        plt.imshow(resized_mask, extent=(x * scale_factor, (x + w) * scale_factor, y * scale_factor, (y + h) * scale_factor), cmap='jet', alpha=0.5)  # Adjust alpha for transparency

    # Set axis limits
    plt.xlim(0, scaled_max_x)
    plt.ylim(0, scaled_max_y)

    # Save the reconstructed image
    plt.axis('off')  # Hide axis
    plt.savefig(os.path.join(save_path, 'reconstructed_image.png'), bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure



################################################################################
##################################### MAIN #####################################
################################################################################

SAVE_DIR = "directory_to_save_predictions"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
# predict_segmentation(best_model, valid_dataset, valid_dataset_vis, SAVE_DIR)
predict_segmentation_colors(best_model, valid_dataset, valid_dataset_vis, SAVE_DIR)
# predict_segmentation_overlay(best_model, valid_dataset, valid_dataset_vis, SAVE_DIR)