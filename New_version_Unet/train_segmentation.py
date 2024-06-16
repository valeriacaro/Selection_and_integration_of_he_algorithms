"""
Train from scratch.
"""

# Libraries
import numpy as np
# import matplotlib.pyplot as plt
import os
import cv2
import json
import random

# from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score

# import seaborn as sns

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Sampler

from dataset import *
from epochs import TrainEpoch, ValidEpoch
import argparse
import wandb
# from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

#####################################################################################
################################## REPRODUCIBILITY ##################################
#####################################################################################


args = argparse.Namespace(
    seed=1234, 
    DATA_DIR = r'path_to_data',
    resolution = 'resolution',
    model_name = 'model_name',
    path_models ='path_to_model',
    path_logs ='path_to_logs'
)

# reinhardYNOUS_x61440_y118784

wandb.login()
log_writer = wandb.init(project="path_to_project", save_code=True, mode='online', name= "basic", config = args)


# General
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

#################################################################################
################################## DIRECTORIES ##################################
#################################################################################

# Directory to the dataset
#DATA_DIR = r'/mnt/gpid08/datasets/ICS/MAMA/TILES/HE/Vall_Hebron/Single resolution 2022/'
#DATA_DIR = r'/mnt/work/users/lauren.jimenez/roi_segmentation/'
#resolution = '10x'
DATA_DIR = os.path.join(args.DATA_DIR, args.resolution)
# images_dir = os.path.join(DATA_DIR, 'Images')
# masks_dir  = os.path.join(DATA_DIR, 'Masks')
# train_files, valid_files = split_dataset(images_dir)

# # Directories with original images and labeled images for train, validation and test.
x_train_dir = os.path.join(DATA_DIR, 'train', 'Images')
y_train_dir = os.path.join(DATA_DIR, 'train', 'Masks')

x_valid_dir = os.path.join(DATA_DIR, 'val', 'Images')
y_valid_dir = os.path.join(DATA_DIR, 'val', 'Masks')

# x_test_dir = os.path.join(DATA_DIR, 'test', 'images')
# y_test_dir = os.path.join(DATA_DIR, 'test', 'masks')

# Directory to save the models
path_models = os.path.join(args.path_models, args.resolution)
path_logs   = os.path.join(args.path_logs, args.resolution)
if not os.path.exists(path_logs):
    os.makedirs(path_logs)
if not os.path.exists(path_models):
    os.makedirs(path_models)

model_name = args.model_name

################################################################################
################################ CONFIGURATION #################################
################################################################################

mean_training = [0.71972791, 0.54526797, 0.68434116]
std_training  = [0.1973122 , 0.23205373, 0.17320338]

ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda'

# CLASSES = ['outside_roi', 'tumor', 'stroma', 'inflammatory infiltration', 'necrosis', 'other']
CLASSES = ['background', 'other', 'tumor', 'insitu']

################################################################################
##################################### DATA #####################################
################################################################################

# Preprocessing
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing_fn.keywords['mean'] = mean_training
preprocessing_fn.keywords['std']  = std_training

# Train and validation datasets
# train_dataset = Dataset(train_files, images_dir, masks_dir,
#                         augmentation=get_training_augmentation(),
#                         preprocessing=get_preprocessing(preprocessing_fn),
#                         classes=CLASSES,
#                         )

# valid_dataset = Dataset(valid_files, images_dir, masks_dir,
#                         preprocessing=get_preprocessing(preprocessing_fn),
#                         classes=CLASSES,
#                         )
train_dataset = Dataset(x_train_dir, y_train_dir,
                        augmentation=get_training_augmentation(),
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES,
                        )

valid_dataset = Dataset(x_valid_dir, y_valid_dir,
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES,
                        )

# Data loader
weights = []
for image, mask in train_dataset:
    weight_image = np.sum((mask.numpy() == 2) | 3 * (mask.numpy() == 3))
    weight = 1.0 if weight_image == 0 else weight_image
    weights.append(weight)

weights = np.array(weights)
weights /= np.sum(weights)

# Define custom sampler
class CustomSampler(Sampler):
    def __init__(self, data_source, weights):
        self.data_source = data_source
        self.weights = weights

    def __iter__(self):
        return iter(torch.multinomial(torch.tensor(self.weights), len(self.data_source), replacement=True))

    def __len__(self):
        return len(self.data_source)

# Create custom sampler
# custom_sampler = CustomSampler(train_dataset, weights)

# Modify DataLoader to use custom sampler
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, worker_init_fn=seed_worker, generator=g, drop_last=True)
# train_loader = DataLoader(train_dataset, batch_size=4, sampler=custom_sampler, num_workers=1, worker_init_fn=seed_worker, generator=g, drop_last=True)

valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1, worker_init_fn=seed_worker, generator=g)


################################################################################
#################################### MODEL #####################################
################################################################################

# Initialize the model
model = smp.Unet(encoder_name=ENCODER,
                 encoder_weights=ENCODER_WEIGHTS,
                 classes=len(CLASSES),
                 activation=None,
                )

# Loss function
weights = torch.tensor([4.87, 2.19, 2.97, 40.65])
# loss = smp.utils.losses.NLLLoss(weight=weights)
loss = nn.CrossEntropyLoss(weight = weights)
# loss = nn.CrossEntropyLoss()
# loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=0)

# Optimizer and scheduler
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001,)])
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Metrics
# ALTRA OPCIO: https://smp.readthedocs.io/en/latest/metrics.html

metrics = {"f1_score": smp.metrics.f1_score,
           "accuracy": smp.metrics.accuracy,
           "recall": smp.metrics.recall,
           "precision": smp.metrics.precision
          }

# metrics = {"f1_score": f1_score,
#            "accuracy": accuracy_score,
#            "recall": recall_score,
#            "precision": precision_score
#           }

################################################################################
#################################### EPOCHS ####################################
################################################################################

train_epoch = TrainEpoch(model,
                         loss=loss,
                         metrics=metrics,
                         optimizer=optimizer,
                         device=DEVICE,
                         verbose=True
)

valid_epoch = ValidEpoch(model,
                         loss=loss,
                         metrics=metrics,
                         device=DEVICE,
                         verbose=True
)


################################################################################
##################################### MAIN #####################################
################################################################################

# Save best model
path_best_model = os.path.join(path_models, model_name)

# Save logs
train_logs_all = {'loss': [], 'f1_score': [], 'recall': [], 'precision': [], 'accuracy': []}
valid_logs_all = {'loss': [], 'f1_score': [], 'recall': [], 'precision': [], 'accuracy': []}
best_valid_logs = {}
path_save_logs = os.path.join(path_logs, 'path_to_logs')

max_score = 0; EPOCHS = 50
for i in range(EPOCHS):
    print('\n Epoch: {}'.format(i))
    
    # Train epoch
    train_logs = train_epoch.run(train_loader)

    # Validation epoch
    valid_logs = valid_epoch.run(valid_loader)
    # scheduler.step()

    logs = {}
    logs['train/loss'] = train_logs['loss']; logs['valid/loss'] = valid_logs['loss']
    logs['train/f1_score'] = train_logs['f1_score']; logs['valid/f1_score'] = valid_logs['f1_score']
    logs['train/recall'] = train_logs['recall']; logs['valid/recall'] = valid_logs['recall']
    logs['train/precision'] = train_logs['precision']; logs['valid/precision'] = valid_logs['precision']
    logs['train/accuracy'] = train_logs['accuracy']; logs['valid/accuracy'] = valid_logs['accuracy']
    logs['train/class_0_accuracy'] = train_logs['class_0_accuracy']; logs['valid/class_0_accuracy'] = valid_logs['class_0_accuracy']
    logs['train/class_1_accuracy'] = train_logs['class_1_accuracy']; logs['valid/class_1_accuracy'] = valid_logs['class_1_accuracy']
    logs['train/class_2_accuracy'] = train_logs['class_2_accuracy']; logs['valid/class_2_accuracy'] = valid_logs['class_2_accuracy']
    logs['train/class_3_accuracy'] = train_logs['class_3_accuracy']; logs['valid/class_3_accuracy'] = valid_logs['class_3_accuracy']
    logs['train/class_0_precision'] = train_logs['class_0_precision']; logs['valid/class_0_precision'] = valid_logs['class_0_precision']
    logs['train/class_1_precision'] = train_logs['class_1_precision']; logs['valid/class_1_precision'] = valid_logs['class_1_precision']
    logs['train/class_2_precision'] = train_logs['class_2_precision']; logs['valid/class_2_precision'] = valid_logs['class_2_precision']
    logs['train/class_3_precision'] = train_logs['class_3_precision']; logs['valid/class_3_precision'] = valid_logs['class_3_precision']
    logs['train/class_0_recall'] = train_logs['class_0_recall']; logs['valid/class_0_recall'] = valid_logs['class_0_recall']
    logs['train/class_1_recall'] = train_logs['class_1_recall']; logs['valid/class_1_recall'] = valid_logs['class_1_recall']
    logs['train/class_2_recall'] = train_logs['class_2_recall']; logs['valid/class_2_recall'] = valid_logs['class_2_recall']
    logs['train/class_3_recall'] = train_logs['class_3_recall']; logs['valid/class_3_recall'] = valid_logs['class_3_recall']
    logs['train/class_0_f1_score'] = train_logs['class_0_f1_score']; logs['valid/class_0_f1_score'] = valid_logs['class_0_f1_score']
    logs['train/class_1_f1_score'] = train_logs['class_1_f1_score']; logs['valid/class_1_f1_score'] = valid_logs['class_1_f1_score']
    logs['train/class_2_f1_score'] = train_logs['class_2_f1_score']; logs['valid/class_2_f1_score'] = valid_logs['class_2_f1_score']
    logs['train/class_3_f1_score'] = train_logs['class_3_f1_score']; logs['valid/class_3_f1_score'] = valid_logs['class_3_f1_score']

    log_writer.log(logs)


    # Save the model
    if max_score < (3*valid_logs['class_2_f1_score'] + 3*valid_logs['class_3_f1_score'] + valid_logs['class_0_f1_score'] + valid_logs['class_1_f1_score'])/4:
        max_score = (3*valid_logs['class_2_f1_score'] + 3*valid_logs['class_3_f1_score'] + valid_logs['class_0_f1_score'] + valid_logs['class_1_f1_score'])/4
        torch.save(model, path_best_model)
        best_valid_logs = valid_logs
        # torch.save(model.state_dict(), path_best_model)
        print('Model saved!  (epoch '+str(i)+') --> recall =', valid_logs['recall'])

    # Save logs
    train_logs_all['loss'].append(train_logs['loss']); valid_logs_all['loss'].append(valid_logs['loss'])
    train_logs_all['f1_score'].append(train_logs['f1_score']); valid_logs_all['f1_score'].append(valid_logs['f1_score'])
    train_logs_all['recall'].append(train_logs['recall']); valid_logs_all['recall'].append(valid_logs['recall'])
    train_logs_all['precision'].append(train_logs['precision']); valid_logs_all['precision'].append(valid_logs['precision'])
    train_logs_all['accuracy'].append(train_logs['accuracy']); valid_logs_all['accuracy'].append(valid_logs['accuracy'])

log_writer.log(best_valid_logs)
# Save logs in a file    
data = {'training': train_logs_all, 'validation': valid_logs_all}
with open(path_save_logs, 'w') as outfile:
    json.dump(data, outfile)

#########
