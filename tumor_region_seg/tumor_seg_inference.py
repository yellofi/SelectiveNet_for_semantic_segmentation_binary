import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import UNet

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        img_list = sorted(os.listdir(self.data_dir))

        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        input = Image.open(os.path.join(self.data_dir, self.img_list[index]))
        input = np.array(input)

        input = input/255.0
        input = input.astype(np.float32)

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'id': self.img_list[index],  'input': input}

        if self.transform:
            data = self.transform(data)

        return data

class ToTensor(object):
    def __call__(self, data):
        input = data['input']

        input = input.transpose((2, 0, 1)).astype(np.float32)

        # data = {'input': torch.from_numpy(input)}
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
        # data = {'input': input}

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


if __name__ == '__main__':

    data_dir = '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/록원재단/AT2/C-MET_slide/patch/S-LC0001-MET/x200'
    batch_size = 1

    print('Load Data...')
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    dataset = Dataset(data_dir=data_dir, transform = transform)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=False)

    rank = 7
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    k_fold = 5
    nets = []

    print('Load Model...')
    for i in range(k_fold):
        print(f'{i+1}-fold - ', end = '')
        ckpt_dir = f'./model/{i+1}-fold/'

        net = UNet().to(device)
        net = net_test_load(ckpt_dir = ckpt_dir, net = net, device=device)

        nets.append(net)

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean
    fn_norm = lambda x : (x-x.min())/(x.max()-x.min())
    fn_classifier = lambda x : 1.0 * (x > 0.5)

    print('Model Inference...')
    results = []
    with torch.no_grad(): 
        net.eval() 
        
        for batch, data in enumerate(loader, 1):
            # forward
            input = data['input'].to(device)
            img_id = data['id']
            
            outputs = []
            for net in nets:
                output = net(input)
                outputs.append(np.squeeze(fn_tonumpy(fn_norm(output)), axis=-1))

            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output_avg = np.mean(np.asarray(outputs), axis = 0)
            final_pred = fn_classifier(output_avg)

            results.append((img_id, input, final_pred, output_avg))
    
    mask_save_dir = './output'
    plot_save_dir = './output/plot'

    try: os.makedirs(mask_save_dir); os.makedirs(plot_save_dir)
    except: pass

    print('Plot and Save...')
    for (img_id, img, pred, output) in results:
        for i in range(batch_size):
            plt.figure(figsize = (20, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(img[i])
            plt.subplot(1, 3, 2)
            plt.imshow(img[i])
            plt.imshow(output[i], cmap ='jet', alpha = 0.3)
            plt.subplot(1, 3, 3)
            plt.imshow(img[i])
            plt.imshow(pred[i], alpha = 0.3)
            plt.savefig(f'{plot_save_dir}/{img_id[i]}_prediction_overlay.jpg', bbox_inches = 'tight')
            plt.close()

            pred_ = Image.fromarray(np.uint8(pred[i]*255))
            pred_.save(f'{mask_save_dir}/{img_id[i]}_predicted_tumor_region.jpg')