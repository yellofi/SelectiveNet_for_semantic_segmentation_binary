import os
import numpy as np
import torch
from PIL import Image
from data_utils import RGB2GH, H_RGB

"""
삼성병원 데이터셋 1차 annotation 

tumor label 기준으로 non_tumorable, tumorable을 구분하여 
5-fold로 해당 파일명들을 (input, label)로 array에 저장해두고  
각각 동일한 비율로 train, valid 구성 

"""

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

class SamsungDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, data_list, patch_mag = 200, patch_size = 256, transform=None, input_type='RGB'):
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
        assert self.input_list[index].split('MET')[0] == self.label_list[index].split('MET')[0], f'image {self.input_list[index]}, label {self.label_list[index]}'

        parent_dir = self.input_list[index].split('MET')[0] + 'MET'

        input = Image.open(os.path.join(self.data_dir, 'patch', parent_dir, f'{self.patch_mag}x_{self.patch_size}', self.input_list[index]))
        label = Image.open(os.path.join(self.data_dir, 'patch', parent_dir, f'{self.patch_mag}x_{self.patch_size}', self.label_list[index])).convert("L")
        
        input, label = np.array(input), np.array(label)
        
        # Blankfield Correction, 밝은 영역 평균을 구해 그걸로 255로 맞추고 scale 다시 맞추는 작업
        # input = correct_background(input)

        input, label = input/255.0, label/255.0
        input, label = input.astype(np.float32), label.astype(np.float32)

        if self.input_type == 'GH':
            input = RGB2GH(input)

        if self.input_type == 'H_RGB':
            input = H_RGB(input)

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {}
        data['id'] = self.input_list[index].split('_input')[0]
        data['input'] = input
        data['label'] = label 

        if self.transform:
            data = self.transform(data)

        return data