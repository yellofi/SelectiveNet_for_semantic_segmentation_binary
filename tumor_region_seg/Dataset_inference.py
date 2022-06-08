import numpy as np
import torch
from PIL import Image
from .data_utils import RGB2GH, H_RGB
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, slide, xy_coord, slide_patch_ratio, input_type = 'RGB', mean = 0.5, std = 0.5):

        self.args = args
        self.slide = slide
        self.xy_coord = xy_coord
        self.slide_level = 0
        self.s_p_ratio = slide_patch_ratio

        self.input_type = input_type
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        coord = self.xy_coord[index]
        input = self.slide.read_region((coord[0], coord[1]), self.slide_level, 
        (self.args.patch_size*self.slide_patch_ratio, self.args.patch_size*self.slide_patch_ratio)).resize(self.args.patch_size, self.args.patch_size)
        input = np.array(input.covert('RGB'))

        # Blankfield Correction, 밝은 영역 평균을 구해 그걸로 255로 맞추고 scale 다시 맞추는 작업
        # input = correct_background(input)
        input = input/255.0
        input = input.astype(np.float32)

        if self.input_type != 'RGB':
            img = input.copy()
            if self.input_type == 'GH':
                input = RGB2GH(input)
            elif self.input_type == 'H_RGB':
                input = H_RGB(input)

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        input = (input - self.mean) / self.std
        input = input.transpose((2, 0, 1)).astype(np.float32)
        input = torch.from_numpy(input)

        if self.input_type != 'RGB':
            return (input, img), coord[0], coord[1]
        else:
            return input, coord[0], coord[1]
