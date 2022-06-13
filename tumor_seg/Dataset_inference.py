import os
import numpy as np
import torch
from PIL import Image
from .data_utils import RGB2GH, H_RGB

class Dataset(torch.utils.data.Dataset):

    def __init__(self, slide, xy_coord, size_on_slide, patch_size, transform=None, mean=0.5, std=0.5, input_type='RGB'):

        self.slide = slide
        self.xy_coord = xy_coord
        self.slide_level = 0
        self.patch_size = patch_size
        self.size_on_slide = size_on_slide

        self.data_dir = '/mnt/ssd1/biomarker/c-met/final_output/check'
        # self.data_dir = '/mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/patch/S-LC0027-MET/200x_1024'

        # img_list = sorted([i for i in os.listdir(self.data_dir) if 'input' in i and 'jpg' in i])

        # self.img_list = img_list

        self.input_type = input_type
        self.transform = transform 
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.xy_coord)
        # return len(self.img_list)

    def __getitem__(self, index):

        # input = Image.open(os.path.join(self.data_dir, self.img_list[index]))
        # input = np.array(input)
        # coord = int(self.img_list[index].split('_')[1]), int(self.img_list[index].split('_')[2])

        coord = self.xy_coord[index]
        input = self.slide.read_region((coord[0], coord[1]), self.slide_level, (self.size_on_slide, self.size_on_slide))
        input = input.resize((self.patch_size, self.patch_size))
        # input.convert('RGB').save(os.path.join(self.data_dir, f'S-LC0027-MET_{coord[0]}_{coord[1]}_input_90.jpg'), quality=90)
        input = np.array(input.convert('RGB'))

        # print(input.dtype, input.shape, input.max(), input.min())
        # Blankfield Correction, 밝은 영역 평균을 구해 그걸로 255로 맞추고 scale 다시 맞추는 작업
        # input = correct_background(input)

        input = input/255.0
        input = input.astype(np.float32)

        data = {}

        if self.input_type != 'RGB':
            img = input.copy()
            data['img'] = img
            if self.input_type == 'GH':
                input = RGB2GH(input)
            elif self.input_type == 'H_RGB':
                input = H_RGB(input)

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        input = (input - self.mean) / self.std
        input = input.transpose((2, 0, 1)).astype(np.float32)
        input = torch.from_numpy(input)

        data['input'] = input
        data['x'] = coord[0]
        data['y'] = coord[1]

        return data
