import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import time
import argparse

from model import *
from data_utils import *
from net_utils import *

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--patch_dir', type=str, 
                        default='*/patch', help='patch directory')
    parser.add_argument('--model_path', type=str, 
                        default='*/model/model.pth', help='model path (*.pth)')
    parser.add_argument('--model_name', type=str, default='0_baseline')

    # parser.add_argument('--save_dir', action="store", type=str,
    #                     default='/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/06_baseline_samsung_data/1-fold/output', help='directory where results would be saved')

    parser.add_argument('--patch_mag', type=int, default = 200)
    parser.add_argument('--patch_size', type=int, default = 1024)

    parser.add_argument('--local_rank', type=int, nargs='+', default=[0], help='local rank')

    parser.add_argument('--input_type', type=str, default='RGB')
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, mean=0.5, std=0.5, input_type = 'RGB'):
        self.data_dir = data_dir
        self.input_type = input_type
        self.mean = mean
        self.std = std

        img_list = sorted([i for i in os.listdir(self.data_dir) if 'input' in i])

        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        input = Image.open(os.path.join(self.data_dir, self.img_list[index]))
        input = np.array(input)

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

        data['id'] = self.img_list[index].split('_input')[0]
        data['input'] = input
        # data = {'id': self.img_list[index],  'input': input}

        return data

def sigmoid(z):
    return 1/(1+np.e**(-(z.astype('float64')-0.5)))

def make_heatmap(output):
    # output = sigmoid(output)
    heatmap = cm.jet(output)[:, :, :3]
    return heatmap.astype('float32')

if __name__ == '__main__':

    args = parse_arguments()

    print('Load Models...')

    rank = args.local_rank
    model_path = args.model_path
    model_path = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_baseline_samsung_data/1-fold/checkpoint/model_epoch197.pth'
    model_path = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/06_baseline_samsung_data/1-fold/checkpoint/model_epoch403.pth'
    input_type = args.input_type

    if len(rank) != 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = UNet(input_type)
        net = net_test_load(model_path, net)
        net = torch.nn.DataParallel(net, device_ids=rank)
        net = net.to(device)
    else:
        # single gpu -> device map location으로 불러와야 gpu 0을 안 씀
        device = torch.device(f'cuda:{rank[0]}')
        net = UNet(input_type).to(device)
        net = net_test_load(model_path, net, device=device) 
        torch.cuda.set_device(rank[0])

    patch_dir = args.patch_dir
    patch_dir = '/mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/patch'
    patch_mag = args.patch_mag
    patch_size = args.patch_size
    
    # target_slides = [27, 32, 47, 59, 80, 87, 90, 94, 106, 107]
    # slide_list = sorted([f for f in os.listdir(patch_dir) if os.path.isdir(os.path.join(patch_dir, f)) and int(f.split('-')[1][2:]) in target_slides])

    slide_list = sorted([f for f in os.listdir(patch_dir) if os.path.isdir(os.path.join(patch_dir, f))])

    print('Inference...')

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean
    fn_norm = lambda x : (x-x.min())/(x.max()-x.min())
    fn_classifier = lambda x : 1.0 * (x > 0.5)

    batch_size = args.batch_size
    model_name = args.model_name
    # model_name = '06_baseline'
    model_name = '06_baseline_403'
    
    total_time = 0

    for slide_id in slide_list:
        
        start_time = time.time()
        data_dir = os.path.join(patch_dir, slide_id, f'{patch_mag}x_{patch_size}')

        dataset = Dataset(data_dir, input_type)
        loader = DataLoader(dataset, batch_size, shuffle=False)

        with torch.no_grad(): 
            net.eval() 
            
            for batch, data in enumerate(loader, 1):
                # forward
                input = data['input'].to(device)
                if input_type == 'GH':
                    img = data['img']
                
                img_id = data['id']
                output = net(input)

                output = np.squeeze(fn_tonumpy(output), axis=-1)
                pred = fn_classifier(output)

                if input_type == 'GH':
                    input = img.to('cpu').detach().numpy()
                else:
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))

                for img_id_, img_, pred_, output_ in zip(img_id, input, pred, output):
  
                    heatmap = make_heatmap(output_)
                    overlay = cv2.addWeighted(img_, 0.7, heatmap, 0.3, 0)
                    overlay = Image.fromarray(np.uint8(overlay*255)).convert('RGB')
                    overlay.save(f'{data_dir}/{img_id_}_{model_name}_heatmap.jpg')

                    pred_ = Image.fromarray(np.uint8(pred_*255)).convert('L')
                    pred_.save(f'{data_dir}/{img_id_}_{model_name}_prediction.png')
        

        taken = time.time() - start_time
        print(f'{slide_id} | #: {len(dataset)} | time: {round(taken, 2)} sec')
        total_time += taken 

    print(f'total time: {round(total_time, 2)} sec')