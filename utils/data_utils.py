import os
import cv2
from PIL import Image
import skimage.color
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

"""
input (output) transformation
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
construct train/valid data list and test data list 
with each 5-fold data list for both tumorable and not_tumorable patches (input, label)
"""
np.random.seed(42)

def split_train_valid(TRAIN_list, valid_ratio = 0.2):
    total_n = len(TRAIN_list)
    valid_idx = np.random.choice(total_n, size = int(total_n*valid_ratio), replace = False)
    train_idx = np.setdiff1d([i for i in range(total_n)], valid_idx)
    return TRAIN_list[train_idx], TRAIN_list[valid_idx]

def construct_train_valid(data_dir, test_fold = 5):
    folds = [1, 2, 3, 4, 5]
    folds.remove(test_fold)

    tumorable_TRAIN, non_tumorable_TRAIN = [], []
    for i in folds:
        tumorable_TRAIN.append(np.load(f'{data_dir}/{i}-fold_tumorable_data.npy'))
        non_tumorable_TRAIN.append(np.load(f'{data_dir}/{i}-fold_non_tumorable_data.npy'))

    tumorable_TRAIN = np.concatenate(tumorable_TRAIN)
    non_tumorable_TRAIN = np.concatenate(non_tumorable_TRAIN)

    t_train, t_valid = split_train_valid(tumorable_TRAIN, 0.2)
    n_train, n_valid = split_train_valid(non_tumorable_TRAIN, 0.2)

    train = np.vstack([t_train, n_train])
    valid = np.vstack([t_valid, n_valid])

    return train, valid

def construct_test(data_dir, test_fold = 1):
    tumorable_test = np.load(f'{data_dir}/{test_fold}-fold_tumorable_data.npy')
    non_tumorable_test = np.load(f'{data_dir}/{test_fold}-fold_non_tumorable_data.npy')

    tumorable_test = np.array(tumorable_test)
    non_tumorable_test = np.array(non_tumorable_test)

    # print(tumorable_test.shape, non_tumorable_test.shape)

    test = np.vstack([tumorable_test, non_tumorable_test])
    return test



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

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 50% 확률로 좌우반전
        if np.random.rand() > 0.5:
            label = np.fliplr(label).copy()
            input = np.fliplr(input)

        # 50% 확률로 상하반전
        if np.random.rand() > 0.5:
            label = np.flipud(label).copy()
            input = np.flipud(input)

        data['input'] = input
        data['label'] = label

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

        return data

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        input = input.transpose((2, 0, 1)).astype(np.float32)

        data['input'] = torch.from_numpy(input)
        data['label'] = torch.from_numpy(label).type(torch.LongTensor)

        return data

"""
PatchDataset requires data_dir, data_list, patch_mag, and patch_size
""" 

class PatchDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, data_list, patch_mag = 200, patch_size = 256, input_type='RGB', transform=None): 
        """
        data_list = [(input_file_1, label_file_1), (input_file_2, label_file_2), ..., (input_file2_N, label_file_N)]

        filename of input_files => {slide_id}_{x_coord}_{y_coord}_input.jpg
        filename of label_files => {slide_id}_{x_coord}_{y_coord}_label.png
        
        input_files, label_files must be in {data_dir}/{patch_mag}x_{patch_size}
        """
        
        self.data_dir = data_dir
        self.data_list = data_list
        self.transform = transform
        self.input_type = input_type
        self.patch_mag = patch_mag
        self.patch_size = patch_size

        input_list, label_list = [], []

        # print(data_list)
        for f in self.data_list:
            # print(f)
 
            assert f[0].split('_input')[0] == f[1].split('_label')[0], f'check the pairness btw input {f[0]} and label {f[1]}'

            input_list.append(f[0])
            label_list.append(f[1])

        self.input_list = input_list
        self.label_list = label_list

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
 
        assert len(self.input_list) == len(self.label_list), f'# of images {len(self.input_list)}, # of labels {len(self.label_list)}' 
        assert self.input_list[index].split('_input')[0] == self.label_list[index].split('_label')[0], f'image {self.input_list[index]}, label {self.label_list[index]}'

        input = Image.open(os.path.join(self.data_dir, f'{self.patch_mag}x_{self.patch_size}', self.input_list[index]))
        label = Image.open(os.path.join(self.data_dir, f'{self.patch_mag}x_{self.patch_size}', self.label_list[index])).convert("L")
        
        input, label = np.array(input), np.array(label)

        input, label = input/255.0, label/255.0
        input, label = input.astype(np.float32), label.astype(np.uint8)

        if self.input_type == 'GH':
            input = RGB2GH(input)
        elif self.input_type == 'H_RGB':
            input = H_RGB(input)

        data = {}
        data['id'] = self.input_list[index].split('_input')[0]
        data['input'] = input
        data['label'] = label 

        if self.transform:
            data = self.transform(data)

        return data




