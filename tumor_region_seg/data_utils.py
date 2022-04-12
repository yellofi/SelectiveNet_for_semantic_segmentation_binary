import os
# import cv2
from PIL import Image
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

# define custom torch dataset for torch dataloader

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        data = sorted(os.listdir(self.data_dir))

        img_list, label_list = [], []
        for f in sorted(data):
            if 'sample' in f and 'S-LC' not in f:
                img_list.append(f) 
            elif 'label' in f and 'S-LC' not in f:
                label_list.append(f)

        self.img_list = img_list
        self.label_list = label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        if len(self.img_list) == len(self.label_list):

            input = Image.open(os.path.join(self.data_dir, self.img_list[index]))
            label = Image.open(os.path.join(self.data_dir, self.label_list[index])).convert("L")
            
            input, label = np.array(input), np.array(label)

            # input = cv2.imread(os.path.join(self.data_dir, self.img_list[index]))
            # label = cv2.imread(os.path.join(self.data_dir, self.label_list[index]), cv2.IMREAD_GRAYSCALE)

            input, label = input/255.0, label/255.0
            input, label = input.astype(np.float32), label.astype(np.float32)

            if label.ndim == 2:
                label = label[:, :, np.newaxis]
            if input.ndim == 2:
                input = input[:, :, np.newaxis]

            data = {'input': input, 'label': label}

            if self.transform:
                data = self.transform(data)

            return data

# preprocssing could be in transforms.Compose

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

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

        data = {'label': label, 'input': input}

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

        data = {'label': label, 'input': input}

        return data

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

# get dataloaders

def create_data_loader(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]: 

    transform_train = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), PartialNonTissue(), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'),transform = transform_train)
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)

    transform_val = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'valid'), transform = transform_val)
    loader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle=True)

    return loader_train, loader_val


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



