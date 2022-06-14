import os
import cv2
from PIL import Image
import skimage.color
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

"""
input transformation
"""
def RGB2GH(rgb_image):
    """
    Arg:
        rgb_image = An Original RGB image, {0, 1}, dtype = np.float32

    Return:
        gh = GH image with 2 channels [g, h], {0, 1}, dtype = np.float32
    """
    g = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) 
    h = skimage.color.separate_stains(rgb_image, skimage.color.hed_from_rgb)[:, :, 0]
    h_min,  h_max = -0.66781543,  1.87798274 
    h = (h - h_min)/(h_max-h_min)
    # h = (h- h.min())/(h.max()-h.min()).astype('float32')
    gh = np.concatenate((g[:, :, np.newaxis], h[:, :, np.newaxis]), axis = -1)
    return gh.astype('float32')

def H_RGB(rgb_image):
    """
    Arg:
        rgb_image = An Original RGB image, {0, 1}, dtype = np.float32

    Return:
        ihc_h = A RGB image of Hematoxyling to the corresponding original rgb image, {0, 1}, dtype = np.float32
    """

    h = skimage.color.separate_stains(rgb_image, skimage.color.hed_from_rgb)[:, :, 0]
    null = np.zeros_like(h)
    ihc_h = skimage.color.combine_stains(np.stack((h, null, null), axis = -1), skimage.color.rgb_from_hed)
    return ihc_h.astype('float32')

"""
preprocssing and augmentation (in transforms.Compose)
"""

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data['input'] = input
        data['label'] = label
        # data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 50% 확률로 좌우반전
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        # 50% 확률로 상하반전
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data['input'] = input
        data['label'] = label
        # data = {'label': label, 'input': input}

        return data

class PartialNonTissue(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        size, size, ch = input.shape
        half_size = size//2 

        non_tissue = np.clip((0.96*np.ones((half_size, half_size, ch)) + \
                             0.005*np.random.randn(half_size, half_size, ch)), 
                             a_min = 0, a_max = 1)
        non_tissue_mask = np.zeros((half_size, half_size, 1))

        if np.random.randint(1, 5) == 1:
            rotation = np.random.randint(1, 5) 
            if rotation == 1:
                input[:half_size, :half_size, :] = non_tissue
                label[:half_size, :half_size, :] = non_tissue_mask
            elif rotation == 2:
                input[:half_size, half_size:, :] = non_tissue
                label[:half_size, half_size:, :] = non_tissue_mask
            elif rotation == 3:
                input[half_size:, :half_size, :] = non_tissue
                label[half_size:, :half_size, :] = non_tissue_mask
            elif rotation == 4:
                input[:half_size, :half_size, :] = non_tissue
                label[:half_size, :half_size, :] = non_tissue_mask

        data['input'] = input
        data['label'] = label

        # data = {'label': label, 'input': input}

        return data

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data['input'] = torch.from_numpy(input)
        data['label'] = torch.from_numpy(label)
        # data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data


# get dataloaders which have their own sampler

# def create_data_loader(data_dir: str, rank: int, world_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]: 

#     transform_train = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

#     dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'),transform = transform_train)
#     sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
#     loader_train = DataLoader(dataset_train, batch_size = batch_size, num_workers=32, sampler = sampler_train, pin_memory=True)

#     transform_val = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

#     dataset_val = Dataset(data_dir=os.path.join(data_dir, 'valid'), transform = transform_val)
#     sampler_val = DistributedSampler(dataset_val, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
#     loader_val = DataLoader(dataset_val, batch_size = batch_size, num_workers=32, sampler = sampler_val, pin_memory=True)

#     return loader_train, loader_val




