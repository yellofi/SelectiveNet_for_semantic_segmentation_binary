import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from data_utils import *
from model import *
from tqdm import tqdm
from typing import OrderedDict
from sklearn.metrics import roc_auc_score 
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import csv

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, 
                        default = './data')
    parser.add_argument('--model_dir', type=str, 
                        default='./model', help='model weights and parameters')
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

    parser.add_argument('--output_scaling', type=str, default=None)
    
    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

def net_test_load(model_dir, net, epoch = 0, device = None):

    ckpt_lst = [i for i in os.listdir(ckpt_dir) if 'pth' in i] # ckpt_dir 아래 있는 모든 파일 리스트를 받아온다
    ckpt_lst.sort(key = lambda f : int(''.join(filter(str.isdigit,f))))

    if device != None:
        dict_model = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[epoch-1]), map_location=device)
    else:
        dict_model = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[epoch-1]), map_location='cpu')

    k = list(dict_model['net'].keys())[0]
    if "module" in k:
        dict_model['net'] = remove_module(dict_model)
    net.load_state_dict(dict_model['net'])

    print('model: ', os.path.join(model_dir, ckpt_lst[epoch-1]))
    return net

def get_target_value_index(array, target_value):
    # index = set([i for i, v in enumerate(array) if v == target_value])
    index = np.where(array == target_value)[0]
    return index

def get_performance(label, output, predict, isprint = False):

    """
    label, output, predict (N, ): numpy ndarray 
    """

    label = label.flatten()
    output = output.flatten()
    predict = predict.flatten()

    C1, C0 = get_target_value_index(label, 1), get_target_value_index(label, 0)
    P1, P0 = get_target_value_index(predict, 1), get_target_value_index(predict, 0)

    TP, TN = np.intersect1d(C1, P1), np.intersect1d(C0, P0) 
    FP, FN = np.setdiff1d(P1, C1), np.setdiff1d(P0, C0)

    accuracy = (len(TP) + len(TN))/(len(C1) + len(C0))

    recall, precision, f1_score = np.NaN, np.NaN, np.NAN
    if len(C1) != 0:    recall = len(TP) / len(C1)
    if len(P1) != 0:    precision = len(TP) / len(P1)

    if recall != np.NaN and precision != np.NaN and recall + precision != 0:
        f1_score = 2*recall*precision/(recall + precision)

    auc_score = np.NaN
    if len(C1) != 0 and len(C0) != 0:
        auc_score = roc_auc_score(label, output)

    if isprint:
        print(f'accuracy: {accuracy:.3f} | recall: {recall:.3f} | precision: {precision:.3f} | f1 score: {f1_score:.3f} | AUC score: {auc_score:.3f}')

    return accuracy, recall, precision, f1_score, auc_score

def sigmoid(z):
    return 1/(1+np.e**(-(z.astype('float64')-0.5)))

def make_heatmap(output):
    # output = output-output.min()
    # output = output/output.max()

    # output = sigmoid(output)
    # output = np.clip(output, 0, 1).astype('float32')
    heatmap = cm.jet(output)[:, :, :3]
    return heatmap.astype('float32')

def save_performance_as_csv(save_dir: str, performance: np.array or list, csv_name: str):

    with open(f'{save_dir}/{csv_name}.csv', 'w', newline ='') as csvfile:
        performance_writer = csv.writer(csvfile, delimiter = '|', quotechar = ' ', quoting=csv.QUOTE_ALL)

        performance_writer.writerow('accuracy| recall| precision| f1 score| AUC score')
        # performance = list(map(str, performance)).join('| ')
        performance = '| '.join(list(map(str, performance)))
        performance_writer.writerow(performance)

if __name__ == '__main__':

    args  = parse_arguments()

    data_dir = args.data_dir
    # data_dir = '/mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/DL_dataset/2205_anno'
    test_fold = args.test_fold
    input_type = args.input_type

    patch_mag = args.patch_mag
    patch_size = args.patch_size

    print(f'Load Test Dataset ({test_fold}-fold)')

    test_list = construct_test(data_dir, test_fold = test_fold)
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset = SamsungDataset(data_dir = data_dir, data_list = test_list, transform = transform, input_type = input_type)

    print("     # of test dataset", len(dataset))
    print("Load Model...")

    rank = args.local_rank
    model_dir = args.model_dir
    
    k_fold = 5
    model_select = [0 for _ in range(k_fold)]
    # model_dir = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/01_5-f_cv_baseline'
    nets = []

    for i in range(k_fold):
        print(f'{i+1}-fold - ', end = '')
 
        # ckpt_dir = f'{model_dir}/{i+1}-fold/checkpoint'
        ckpt_dir = f'{model_dir}/{i+1}-fold'

        if len(rank) != 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = UNet(input_type, DataParallel=True)
            net = net_test_load(ckpt_dir, net, epoch = model_select[i])
            net = torch.nn.DataParallel(net, device_ids=rank)
            net = net.to(device)
        else:
            # single gpu -> device map location으로 불러와야 gpu 0을 안 씀
            device = torch.device(f'cuda:{rank[0]}')
            net = UNet(input_type).to(device)
            net = net_test_load(ckpt_dir, net, epoch = model_select[i], device=device) 

        nets.append(net)

    if len(rank) == 1:
        torch.cuda.set_device(rank[0])
    
    print("Model Inference...")

    batch_size = args.batch_size # number of samples in an WSI (or a source image)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=False)

    output_scale = args.output_scaling
    # output_scale = 'clip'
    
    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean

    fn_scale_minmax = lambda x : (x-x.min())/(x.max()-x.min())
    fn_sigmoid = lambda x : 1/(1+ np.exp(-(x.astype('float64')-0.5)))
    fn_clip = lambda x: np.clip(x, 0, 1).astype('float32')

    fn_classifier = lambda x : 1.0 * (x > 0.5)
    
    results = []
    with torch.no_grad(): 
        net.eval() 
        loss_arr = []

        for data in tqdm(loader, total = len(loader)):
            # forward
            inputs = data['input'].to(device)
            labels = data['label'].to(device)
            if input_type == 'GH':
                img = data['img']

            outputs = []
            for net in nets:
                output = net(inputs)
                output = np.squeeze(fn_tonumpy(output), axis=-1)
                if output_scale == 'minmax':
                    output = fn_scale_minmax(output)
                elif output_scale == 'clip':
                    output = np.clip(output, 0, 1)
                elif output_scale == 'sigmoid':
                    output = fn_sigmoid(output)

                outputs.append(output)

            outputs_avg = np.mean(np.asarray(outputs), axis = 0)
            final_preds = fn_classifier(outputs_avg)

            ids = data['id']
            inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
            labels = np.squeeze(fn_tonumpy(labels), axis=-1)
            results.append((ids, inputs, labels, outputs_avg, final_preds))

    save_dir = args.save_dir
    # save_dir = '/mnt/ssd1/biomarker/c-met/tumor_seg/model/01_5-f_cv_baseline/clip'
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