
# Libraries
import numpy as np
import cv2
import os

import albumentations as albu
import torch
from torch.utils.data import Dataset as BaseDataset
import torchstain
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


#################################################################################
################################# SPLIT DATASET #################################
#################################################################################

def split_dataset(images_path, seed=42, train_prop=0.8):
    
    # Files names
    allFileNames = os.listdir(images_path)

    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(allFileNames)

    # Split train and validation
    train_FileNames, valid_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*train_prop)])
    train_FileNames, valid_FileNames = train_FileNames.tolist(), valid_FileNames.tolist()
    
    # # Train filenames
    # train_FileNames = [name for name in train_FileNames.tolist()]
    # valid_FileNames = [name for name in valid_FileNames.tolist()]

    return train_FileNames, valid_FileNames


#################################################################################
################################# AUGMENTATION ##################################
#################################################################################

def get_training_augmentation():
    train_transform = [
        
        albu.Flip(p=0.25),
        albu.HorizontalFlip(p=0.25),
        albu.VerticalFlip(p=0.25),
        albu.RandomRotate90(p=0.5),

        albu.OneOf([
            albu.GaussNoise(p=0.5, var_limit=(10,100)),
            albu.GaussianBlur(p=0.5, blur_limit=(3, 3)),
            albu.MedianBlur(p=0.5, blur_limit=(5, 5)),
            
            albu.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0),
            albu.RandomBrightnessContrast(p=0.5, contrast_limit=0.05, brightness_limit=0),
            # albu.RandomGamma(p=0.5, gamma_limit=(1, 1)),

        ],p=0.5)       

        # albu.RandomBrightnessContrast(p=0.2),
    ]
    return albu.Compose(train_transform)


#################################################################################
################################# PREPROCESSING #################################
#################################################################################

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_tensor_mask(x, **kwargs):
    if x.ndim == 3:
        return torch.tensor(x.transpose(2, 0, 1), dtype=torch.long)
    return torch.tensor(x, dtype=torch.long)

def to_tensor_mask_oh(x, **kwargs):
    return torch.tensor(x.transpose(2, 0, 1), dtype=torch.long)

def get_preprocessing():
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albu.Lambda(image=to_tensor, mask=to_tensor_mask),
    ]
    return albu.Compose(_transform)

#################################################################################
#################################### DATASET ####################################
#################################################################################

# target_image = cv2.cvtColor(cv2.imread(r'path_to_target_image'), cv2.COLOR_BGR2RGB)
# T = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Lambda(lambda x: x*255)
# ])
# torch_normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
# torch_normalizer.fit(T(target_image))

class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    # CLASSES = ['all', 'outside_roi', 'tumor', 'stroma', 'inflammatory infiltration', 'necrosis', 'other']
    CLASSES = ['background', 'other', 'tumor', 'insitu']
    ind2lab = [0,1,2,3]

    def __init__(
            self,
            path,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        
        # Images and masks
        # self.ids = ids
        self.data_df = pd.read_csv(path, names=['Indice', 'Ruta', 'Clase'], header=0)

        # Convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        # Augmentation and preprocessing
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    
    def __getitem__(self, i):

        # Read data (image and mask)
        ruta_de_la_imagen = self.data_df.loc[i, 'Ruta']
        image = cv2.imread(ruta_de_la_imagen)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        clase = self.data_df.loc[i, 'Clase']

        # Data augmentation
        if self.augmentation:

            # if np.any((mask == 2) | (mask ==3)):

            sample = self.augmentation(image=image)
            image = sample['image']
            
        # Preprocessing
        if self.preprocessing:

            # image = torch_normalizer.normalize(T(image))

            # image = image.cpu().numpy()

            image = image//255
            pil_image = Image.fromarray(image, 'RGB')
            image = self.preprocessing(pil_image)

        return image, clase
    

    def __len__(self):
        return len(self.data_df)