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
    parser.add_argument('--model_dir', type=str, 
                        default='./model', help='model directory (5 models corresponing to each fold)')

    parser.add_argument('--local_rank', type=int, default=7 , help='local rank (gpu idx)')
    
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

        img_list = sorted(os.listdir(self.data_dir))

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

        data['id'] = self.img_list[index]
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

def net_test_load(ckpt_dir, net, epoch = 0, device = torch.device('cuda:0')):
    if not os.path.exists(ckpt_dir): # 저장된 네트워크가 없다면 인풋을 그대로 반환
        epoch = 0
        return net
    
    ckpt_lst = os.listdir(ckpt_dir) # ckpt_dir 아래 있는 모든 파일 리스트를 받아온다
    ckpt_lst.sort(key = lambda f : int(''.join(filter(str.isdigit,f))))

    print(f'{ckpt_lst[epoch-1]}')
    dict_model = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[epoch-1]), map_location=device)
    
    net.load_state_dict(dict_model['net'])

    return net

def sigmoid(z):
    return 1/(1+np.e**(-z))

def make_heatmap(output):
    output = sigmoid(output)
    heatmap = cm.jet(output)[:, :, :3]
    return heatmap.astype('float32')


if __name__ == '__main__':

    args = parse_arguments()

    print('Load Models...')
    
    rank = args.local_rank
    model_dir = args.model_dir
    batch_size = args.batch_size
    input_type = args.input_type
    patch_dir = args.patch_dir
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model_select = []

    model_dir = '/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/01_5-f_cv_baseline'
    model_select = [207, 208, 263, 290, 285] 

    # model_dir = '/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/05_5-f_cv_GH'
    # model_select = [124, 114, 132, 103, 107] # 200 epoch
  
    k_fold = 5
    nets = []

    for i in range(k_fold):
        print(f'{i+1}-fold - ', end = '')
 
        ckpt_dir = f'{model_dir}/{i+1}-fold/checkpoint'

        net = UNet(input_type).to(device)
        # print(net)
        
        if len(model_select) == k_fold:
            net = net_test_load(ckpt_dir = ckpt_dir, net = net, epoch = model_select[i], device=device)
        else: 
            net = net_test_load(ckpt_dir = ckpt_dir, net = net, epoch = 0, device=device)

        nets.append(net)

    slide_list = sorted([f for f in os.listdir(patch_dir) if os.path.isdir(os.path.join(patch_dir, f))])

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean
    fn_norm = lambda x : (x-x.min())/(x.max()-x.min())
    fn_classifier = lambda x : 1.0 * (x > 0.5)
    
    print('Inference...')

    total_time = 0
    for slide_id in slide_list:
        
        start_time = time.time()
        data_dir = os.path.join(patch_dir, slide_id, '200x')
        mask_save_dir = f'{model_dir}/output/slide/{slide_id}'

        try: os.makedirs(mask_save_dir)
        except: pass
        
        transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
        # dataset = Dataset(data_dir=data_dir, transform = transform)
        dataset = Dataset(data_dir, transform, input_type)
        loader = DataLoader(dataset, batch_size = batch_size, shuffle=False)

        results = []
        with torch.no_grad(): 
            net.eval() 
            
            for batch, data in enumerate(loader, 1):
                # forward
                input = data['input'].to(device)
                if input_type == 'GH':
                    img = data['img']
                
                img_id = data['id']
                
                outputs = []
                for net in nets:
                    output = net(input)
                    outputs.append(np.squeeze(fn_tonumpy(fn_norm(output)), axis=-1))
                    # outputs.append(np.squeeze(fn_tonumpy(output), axis=-1))

                if input_type == 'GH':
                    input = img.to('cpu').detach().numpy()
                    # print(input.shape, input.dtype, input.max(), input.min())
                else:
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))

                output_avg = np.mean(np.asarray(outputs), axis = 0)
                final_pred = fn_classifier(output_avg)

                for img_id_, img_, pred_, output_ in zip(img_id, input, final_pred, output_avg):
  
                    heatmap = make_heatmap(output_)
                    overlay = cv2.addWeighted(img_, 0.7, heatmap, 0.3, 0)
                    overlay = Image.fromarray(np.uint8(overlay*255))
                    overlay.save(f'{mask_save_dir}/{img_id_[:-4]}_heatmap_overlay.jpg')

                    pred_ = Image.fromarray(np.uint8(pred_*255))
                    pred_.save(f'{mask_save_dir}/{img_id_[:-4]}_predicted_tumor_region.png')

        taken = time.time() - start_time
        print(f'{slide_id} | #: {len(os.listdir(data_dir))} | time: {round(taken, 2)} sec')
        total_time += taken 

    print(f'total time: {round(total_time, 2)} sec')