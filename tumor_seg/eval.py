import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import csv

from data_utils import *
from Dataset_samsung import *
from model import UNet
from net_utils import *
from compute_metric import *

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, 
                        default = './data')
    parser.add_argument('--model_path', type=str, 
                        default='./model.pth', help='model weights and parameters')
    parser.add_argument('--save_dir', type=str, 
                        default='./output', help='saving results')

    parser.add_argument('--local_rank', type=int, nargs='+', default=[0], help='single gpu or DP')

    parser.add_argument('--test_fold', type = int, 
                        default = 1, help = 'which fold in 5-fold cv')
    
    parser.add_argument('--data_type', type=str, default = 'samsung', 
                        help='sample or samsung')

    parser.add_argument('--patch_mag', type=int, default = 200)
    parser.add_argument('--patch_size', type=int, default = 256)

    parser.add_argument('--input_type', type=str, default='RGB')
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

def sigmoid(z):
    return 1/(1+np.e**(-(z.astype('float64')-0.5)))

def make_heatmap(output):
    # output = sigmoid(output)
    heatmap = cm.jet(output)[:, :, :3]
    return heatmap.astype('float32')

def save_performance_as_csv(save_dir: str, performance: np.array or list, csv_name: str):

    with open(f'{save_dir}/{csv_name}.csv', 'w', newline ='') as csvfile:
        performance_writer = csv.writer(csvfile, delimiter = '| ', quotechar = ' ', quoting=csv.QUOTE_ALL)

        performance_writer.writerow('accuracy| recall| precision| f1 score| AUC score')
        performance = list(map(str, performance)).join('| ')
        performance_writer.writerow(performance)

if __name__ == '__main__':

    args  = parse_arguments()

    data_dir = args.data_dir
    test_fold = args.test_fold
    input_type = args.input_type

    patch_mag = args.patch_mag
    patch_size = args.patch_size

    print(f'Load Test Dataset ({test_fold}-fold)')

    test_list = construct_test(data_dir, test_fold = test_fold)
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    dataset = SamsungDataset(data_dir = data_dir, data_list = test_list, transform = transform, input_type = input_type)

    print("     # of test dataset", len(dataset))

    rank = args.local_rank
    model_path = args.model_path
    n_cls = args.n_cls

    print("Load Model...")
    
    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean
    fn_classifier = lambda x : 1.0 * (x > 0.5)

    if args.model_arch == 'UNet':
        net = UNet(input_type, n_cls, selective=args.selective)
 

    if len(rank) != 1: # gpu 여러개 쓰고 싶을때, DP 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif len(rank) == 1: # 0번 말고 다른 gpu도 single로 쓰고 싶을때
        device = torch.device(f'cuda:{rank}')
        net = net.to(device)

    if not os.path.exists(model_path): 

        if len(rank) != 1:
            ckpt = torch.load(model_path, map_location='cpu')
        elif len(rank) == 1:
            ckpt = torch.load(model_path, map_location=device)

        try: ckpt['net'] = remove_module(ckpt)
        except: pass

        net.load_state_dict(ckpt['net'])

    if len(rank) > 1:
        net = torch.nn.DataParallel(net, device_ids=rank)
        net = net.cuda()

    # if len(rank) != 1:
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     net = UNet(input_type)
    #     net = net_test_load(model_path, net)
    #     net = torch.nn.DataParallel(net, device_ids=rank)
    #     net = net.to(device)
    # else:
    #     # single gpu -> device map location으로 불러와야 gpu 0을 안 씀
    #     device = torch.device(f'cuda:{rank[0]}')
    #     net = UNet(input_type).to(device)
    #     net = net_test_load(model_path, net, device=device) 
    #     torch.cuda.set_device(rank[0])
        
    batch_size = args.batch_size # number of samples in an WSI (or a source image)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=False)
    
    print("Model Inference...")
    results = []
    with torch.no_grad(): 
        net.eval() 
        loss_arr = []

        for data in tqdm(loader, total = len(loader)):
            # forward
            
            labels = data['label'].to(device)
            inputs = data['input'].to(device)
            outputs = net(inputs)
            
            ids = data['id']
            inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
            labels = np.squeeze(fn_tonumpy(labels), axis=-1)
            preds = np.squeeze(fn_tonumpy(fn_classifier(outputs)), axis=-1)
            probs = np.squeeze(fn_tonumpy(outputs), axis=-1)
            results.append((ids, inputs, labels, probs, preds))

    save_dir = args.save_dir
    result_dir = os.path.join(save_dir, 'model_pred')
    os.makedirs(result_dir, exist_ok = True)

    print("Measure Performance and Save results")

    patch_level_performance =  []
    for b, (ids, inputs, labels, outputs, preds) in tqdm(enumerate(results), total = len(results)):
        for j, (id, i, l, o, p) in enumerate(zip(ids, inputs, labels, outputs, preds)): 
            # save_dir = os.path.join(data_dir, 'patch', parent_dir, f'{patch_mag}x_{patch_size}')

            # save_dir = os.path.join(result_dir, slide_name)

            heatmap = make_heatmap(o)
            overlay = cv2.addWeighted(i, 0.7, heatmap, 0.3, 0)

            input = Image.fromarray(np.uint8(i*255)).convert('RGB')
            label = Image.fromarray(np.uint8(l*255)).convert('L')
            overlay = Image.fromarray(np.uint8(overlay*255)).convert('RGB')
            pred = Image.fromarray(np.uint8(p*255)).convert('L')

            input.save(os.path.join(result_dir, f'{id}_input.jpg'))
            label.save(os.path.join(result_dir, f'{id}_label.png'))
            overlay.save(os.path.join(result_dir, f'{id}_heatmap_overlay.jpg'))
            pred.save(os.path.join(result_dir, f'{id}_pred.png'))

            patch_level_performance.append(get_performance(l, o, p))

    patch_level_performance = np.nanmean(np.concatenate([patch_level_performance]), axis = 0)

    print(f'patch-level average | accuracy: {patch_level_performance[0]:.3f} | recall: {patch_level_performance[1]:.3f} | precision: {patch_level_performance[2]:.3f} | f1 score: {patch_level_performance[3]:.3f} | AUC score: {patch_level_performance[4]:.3f}')
    
    save_performance_as_csv(save_dir=save_dir, performance=patch_level_performance, csv_name = "input-level_average_performance")