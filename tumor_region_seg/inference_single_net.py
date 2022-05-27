import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from model import *
from data_utils import *
import time
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--patch_dir', type=str, 
                        default='/mnt/hdd1/c-MET_datasets/SLIDE_DATA/록원재단/AT2/C-MET_slide/patch_on_ROI/sobel+blurrity_check', help='patch directory')
    parser.add_argument('--model_path', type=str, 
                        default='/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/06_baseline_samsung_data/1-fold/checkpoint/model_epoch159.pth', help='model path (*.pth)')
    parser.add_argument('--save_dir', action="store", type=str,
                        default='/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/06_baseline_samsung_data/1-fold/output', help='directory where results would be saved')

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

    def __init__(self, data_dir, transform=None, input_type = 'RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.input_type = input_type

        img_list = sorted([i for i in os.listdir(self.data_dir) if 'input' in i])

        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        input = Image.open(os.path.join(self.data_dir, self.img_list[index]))
        input = np.array(input)

        # Blankfield Correction, 밝은 영역 평균을 구해 그걸로 255로 맞추고 scale 다시 맞추는 작업
        # input = correct_background(input)
        input = input/255.0
        input = input.astype(np.float32)

        data = {}
        if self.input_type == 'GH':
            img = input.copy()
            data['img']= img

            input = RGB2GH(input)

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data['id'] = self.img_list[index].split('_input')[0]
        data['input'] = input
        # data = {'id': self.img_list[index],  'input': input}

        if self.transform:
            data = self.transform(data)

        return data

class ToTensor(object):
    def __call__(self, data):
        input = data['input']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        data['input'] = torch.from_numpy(input)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input = data['input']
        input = (input - self.mean) / self.std
        data['input'] = input
        return data

def net_test_load(model_path, net, device=None):
    if device != None:
        dict_model = torch.load(model_path, map_location=device)
    else:
        dict_model = torch.load(model_path, map_location='cpu')

    k = list(dict_model['net'].keys())[0]
    if "module" in k:
        dict_model['net'] = remove_module(dict_model)
    net.load_state_dict(dict_model['net'])

    print('     model: ', model_path)
    return net

def sigmoid(z):
    return 1/(1+np.e**(-(z.astype('float64')-0.5)))

def make_heatmap(output):
    # output = output-output.min()
    # output = output/output.max()

    output = sigmoid(output)
    # output = np.clip(output, 0, 1).astype('float32')
    heatmap = cm.jet(output)[:, :, :3]
    return heatmap.astype('float32')

if __name__ == '__main__':

    args = parse_arguments()

    print('Load Models...')

    rank = args.local_rank
    model_path = args.model_path
    input_type = args.input_type

    if len(rank) != 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = UNet(input_type, DataParallel=True)
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
    patch_mag = args.patch_mag
    patch_size = args.patch_size
    
    target_slides = [27, 32, 47, 59, 80, 87, 90, 94, 106, 107]
    
    slide_list = sorted([f for f in os.listdir(patch_dir) if os.path.isdir(os.path.join(patch_dir, f)) and int(f.split('-')[1][2:]) in target_slides])

    # slide_list = sorted([f for f in os.listdir(patch_dir) if os.path.isdir(os.path.join(patch_dir, f))])

    print('Inference...')

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean
    fn_norm = lambda x : (x-x.min())/(x.max()-x.min())
    fn_classifier = lambda x : 1.0 * (x > 0.5)

    save_dir = args.save_dir
    batch_size = args.batch_size
    
    total_time = 0

    for slide_id in slide_list:
        
        start_time = time.time()
        data_dir = os.path.join(patch_dir, slide_id, f'{patch_mag}x_{patch_size}')
        # mask_save_dir = f'{save_dir}/slide/{slide_id}'
        # plot_save_dir = mask_save_dir + '/plot'

        # try: os.makedirs(mask_save_dir)
        # except: pass
        
        transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
        dataset = Dataset(data_dir, transform, input_type)
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

                # pred = np.squeeze(fn_tonumpy(fn_classifier(output)), axis=-1)
                output = np.squeeze(fn_tonumpy(output), axis=-1)
                pred = fn_classifier(output)
                # pred = fn_classifier(output).astype('float32')

                if input_type == 'GH':
                    input = img.to('cpu').detach().numpy()
                else:
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))

                for img_id_, img_, pred_, output_ in zip(img_id, input, pred, output):
  
                    heatmap = make_heatmap(output_)
                    overlay = cv2.addWeighted(img_, 0.7, heatmap, 0.3, 0)
                    overlay = Image.fromarray(np.uint8(overlay*255)).convert('RGB')
                    overlay.save(f'{data_dir}/{img_id_}_06_baseline_heatmap.jpg')

                    pred_ = Image.fromarray(np.uint8(pred_*255)).convert('L')
                    pred_.save(f'{data_dir}/{img_id_}_06_baseline_prediction.png')
        

        taken = time.time() - start_time
        print(f'{slide_id} | #: {len(os.listdir(data_dir))} | time: {round(taken, 2)} sec')
        total_time += taken 

    print(f'total time: {round(total_time, 2)} sec')