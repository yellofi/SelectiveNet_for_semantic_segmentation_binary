import os
import numpy as np
import torch
from PIL import Image
from data_utils import *

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, input_type='RGB', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.input_type = input_type

        data = sorted(os.listdir(self.data_dir))

        input_list, label_list = [], []
        for f in sorted(data):
            if 'sample' in f:
            # if 'sample' in f and 'S-LC' not in f:
                input_list.append(f)
            elif 'label' in f: 
            # elif 'label' in f and 'S-LC' not in f:
                label_list.append(f)

        self.input_list = input_list
        self.label_list = label_list

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):

        if len(self.input_list) == len(self.label_list):

            input = Image.open(os.path.join(self.data_dir, self.input_list[index]))
            label = Image.open(os.path.join(self.data_dir, self.label_list[index])).convert("L")
            
            input, label = np.array(input), np.array(label)
            
            # Blankfield Correction, 밝은 영역 평균을 구해 그걸로 255로 맞추고 scale 다시 맞추는 작업
            # input = correct_background(input)

            input, label = input/255.0, label/255.0
            input, label = input.astype(np.float32), label.astype(np.uint8)

            if self.input_type == 'GH':
                input = RGB2GH(input)
            elif self.input_type == 'H_RGB':
                input = H_RGB(input)

            # if label.ndim == 2:
            #     label = label[:, :, np.newaxis]
            # if input.ndim == 2:
            #     input = input[:, :, np.newaxis]

            data = {'input': input, 'label': label}

            if self.transform:
                data = self.transform(data)

            return data