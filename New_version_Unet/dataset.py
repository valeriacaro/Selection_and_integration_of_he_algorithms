
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
import stain_utils as utils
import stainNorm_Vahadane


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

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor_mask),
    ]
    return albu.Compose(_transform)

#################################################################################
#################################### DATASET ####################################
#################################################################################

# target_image = cv2.cvtColor(cv2.imread(r'target'), cv2.COLOR_BGR2RGB)
# T = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Lambda(lambda x: x*255)
# ])
# FOR REINHARD
# torch_normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
# torch_normalizer.fit(T(target_image))
# FOR MACENKO
# torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
# torch_normalizer.fit(T(target_image))
# FOR VAHADANE
# n=stainNorm_Vahadane.Normalizer()
# n.fit(target_image)

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
            # ids,
            images_dir, masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        
        # Images and masks
        # self.ids = ids
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        # Augmentation and preprocessing
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    
    def __getitem__(self, i):

        # Read data (image and mask)
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Labels in the mask
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = mask//85

        # Mask as one-hot
        masks = [(mask == self.ind2lab[v]) for v in self.class_values]
        mask_oh = np.stack(masks, axis=-1).astype('float')

        # Data augmentation
        if self.augmentation:

            # if np.any((mask == 2) | (mask ==3)):

            sample = self.augmentation(image=image, masks=[mask, mask_oh])
            image, mask, mask_oh = sample['image'], sample['masks'][0], sample['masks'][1]
            
        # Preprocessing
        if self.preprocessing:

            # FOR REINHARD
            # image = torch_normalizer.normalize(T(image))
            # FOR MACENKO
            # image, H, E = torch_normalizer.normalize(T(image))
            # FOR VAHADANE
            # image = n.transform(image)

            # image = image.cpu().numpy()

            image = image//255

            sample = self.preprocessing(image=image, masks=[mask, mask_oh])
            image, mask, mask_oh = sample['image'], sample['masks'][0], sample['masks'][1]

        return image, mask

    def __len__(self):
        return len(self.ids)