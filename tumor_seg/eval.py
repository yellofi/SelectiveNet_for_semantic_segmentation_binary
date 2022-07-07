import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
# import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import csv

from data_utils import *
from Dataset_sample import Dataset
from Dataset_samsung import *
from model import *
from net_utils import *
from compute_metric import Evaluator

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, 
                        default = './data')

    parser.add_argument('--test_fold', type = int, 
                        default = 1, help = 'which fold in 5-fold cv')
    parser.add_argument('--data_type', type=str, default = 'samsung', 
                        help='sample or samsung')

    parser.add_argument('--input_type', type=str, default='RGB')
    parser.add_argument('--patch_mag', type=int, default = 200)
    parser.add_argument('--patch_size', type=int, default = 256)
    parser.add_argument('--n_cls', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16, help='Dataloader num_workers')

    parser.add_argument('--model_dir', type=str, 
                        default='*/model', help='network ckpt (.pth) directory')
    parser.add_argument('--model_arch', type=str, nargs = '+',
                        default = ['UNet_B'], choices=['UNet_B'])
    parser.add_argument('--selective', type=bool, default = False, help = 'Is the network based on SelectiveNet?')
    parser.add_argument('--select_eval', type=bool, default = False, help = 'calculate metrics with/without selection')
    parser.add_argument('--output_dim', type=str, default = 'NHW', choices=['NCHW', 'NHW'])

    parser.add_argument('--single_scale', type=str, default = 'sigmoid', choices=['None', 'clip', 'sigmoid', 'minmax'])
    parser.add_argument('--ens_scale', type=str,default = 'None', choices=['None', 'clip', 'sigmoid', 'minmax'])

    parser.add_argument('--cut_off', type=float, default=0.5, help = 'prob > cut_off -> pred: 1')
    parser.add_argument('--s_cut_off', type=float, default=0.5, help = 'selection > cut_off -> select: 1')

    parser.add_argument('--local_rank', type=int, nargs='+', default=[0], help='local gpu ids')
    parser.add_argument('--info_print', type=bool, default = False)

    parser.add_argument('--save_dir', type=str, 
                        default='./output', help='saving results')
    
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
    batch_size = args.batch_size
    num_workers = args.num_workers

    if args.info_print:
        print(f'Load Test Dataset ({test_fold}-fold)')

    test_list = construct_test(data_dir, test_fold = test_fold)
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    if args.data_type == 'samsung':
        test_set = SamsungDataset(data_dir = data_dir, data_list = test_list, transform = transform, input_type = input_type)
    elif args.data_type == 'sample':
        data_dir = f'{args.data_dir}/{args.fold}-fold'
        test_set = Dataset(data_dir=os.path.join(data_dir, 'valid'), input_type=input_type, transform=transform) 


    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=False)

    if args.info_print:
        print(f'    patch mag: {patch_mag}')
        print(f'    patch size: {patch_size}')
        print(f'    batch size: {batch_size}')
        print(f'    num workers: {num_workers}')
        print("     # of test dataset", len(test_set))

    """
    Load Tumor Segmentation Model
    """

    if args.info_print:
        print("Load Tumor Segmentation Model...")

    rank = args.local_rank
    model_dir = args.model_dir
    input_type = args.input_type
    model_arch = args.model_arch
    selective = args.selective 

    model_list = sorted([ckpt for ckpt in os.listdir(model_dir) if 'pth' in ckpt])
    
    if len(model_list) != 1 and len(model_arch) == 1:
        model_arch_ = model_arch[0]
        model_arch = [model_arch_ for _ in range(len(model_list))]

    model_dict = {'UNet_B': UNet_B}
    nets = []
    
    for i in range(len(model_list)):
 
        model_path = os.path.join(model_dir, model_list[i])
        if args.info_print:
            print(f'    {model_path} - {model_arch[i]} / SelectiveNet: {selective}')

        net = model_dict[model_arch[i]](input_type, selective=selective)
 
        # define a device 
        if len(rank) != 1: # gpu 여러개 쓰고 싶을때, DP 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif len(rank) == 1: # 0번 말고 다른 gpu도 single로 쓰고 싶을때
            device = torch.device(f'cuda:{rank}')
            net = net.to(device)

        # net = net_test_load(model_path, net, device=device)
        if len(rank) != 1:
            ckpt = torch.load(model_path, map_location='cpu')
        elif len(rank) == 1:
            ckpt = torch.load(model_path, map_location=device)

        try: ckpt['net'] = remove_module(ckpt)
        except: pass

        net.load_state_dict(ckpt['net'])

        # using multi-gpu via DataParallel
        if len(rank) > 1:
            net = torch.nn.DataParallel(net, device_ids=rank)
            net = net.cuda() # == net.to(device) 
        
        net.train(False)
        nets.append(net)

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

    print("Model Prediction...")

    # converting functions
    NCHW_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
    NHW_tonumpy = lambda x: x.to('cpu').detach().numpy()
    fn_denorm = lambda x, mean, std : (x*std) + mean

    if args.output_dim == 'NCHW':
        fn_tonumpy = lambda x: np.squeeze(NCHW_tonumpy(x), axis=-1)
    elif args.output_dim == 'NHW':
        fn_tonumpy = lambda x: NHW_tonumpy(x)

    # output scaling
    single_scale = args.single_scale
    ensemble_scale = args.ens_scale
    fn_scale_minmax = lambda x : (x-x.min())/(x.max()-x.min())
    fn_sigmoid = lambda x : 1/(1+ np.exp(-x))
    fn_clip = lambda x: np.clip(x, 0, 1)
    
    # probability cut-off for classification
    fn_classifier = lambda x, cut_off : 1.0 * (x > cut_off)

    select_eval = args.select_eval

    if select_eval:
        total, total_reject = 0, 0

    # results = []

    # measuring class
    evaluator = Evaluator(num_class=args.n_cls, selective=args.select_eval)

    with torch.no_grad(): 
        # net.eval()
        # loss_arr = []

        for data in tqdm(test_loader, total = len(test_loader)):
            # forward
            input = data['input'].to(device)
            label = data['label'].to(device) # (N, H, W)

            # single model
            if len(nets) == 1:
                if not selective:
                    output = nets[0](input) # (N, H, W) or (N, 2, H, W)
                else:
                    output, selection, _ = nets[0](input) # (N, H, W) or (N, 2, H ,W)
                output = fn_tonumpy(output)

            # ensemble, selective 불가
            else:
                outputs = []
                for net in nets:
                    output = net(input)
                    if ensemble_scale == 'None':
                        outputs.append(fn_tonumpy(output))
                    elif ensemble_scale == 'clip':
                        outputs.append(fn_tonumpy(fn_clip(output)))
                    elif ensemble_scale == 'minmax':
                        outputs.append(fn_tonumpy(fn_scale_minmax(output)))
                    elif ensemble_scale == 'sigmoid':
                        outputs.append(fn_tonumpy(fn_sigmoid(output)))
                output = np.mean(np.asarray(outputs), axis = 0)
                del outputs
            
            ids = data['id']
            input = NCHW_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            label = NHW_tonumpy(label).astype('uint8')

            if len(output.shape) == 4:
                pred = np.argmax(output, axis = -1).astype('uint8')
            elif len(output.shape) == 3:
                if single_scale == 'sigmoid':
                    output = fn_sigmoid(output)
                pred = fn_classifier(output, cut_off=args.cut_off).astype('uint8')

            if select_eval:
                selection = fn_tonumpy(selection)
                if len(selection.shape) == 4:
                    # _, selection = torch.max(selection, dim=1)
                    selection = np.argmax(selection, -1).astype('uint8')
                elif len(selection.shape) == 3:
                    if single_scale == 'sigmoid':
                        selection = fn_sigmoid(selection)
                    selection = fn_classifier(selection, cut_off=args.s_cut_off)

                total += output.size
                reject = output.size - selection.sum().item()
                total_reject += reject

                evaluator.add_batch(label, pred, selection=selection)
            else:
                evaluator.add_batch(label, pred)

            del input, output, pred
            if selective:
                del selection

            torch.cuda.empty_cache()

            # results.append((ids, input, label, output, pred))

    CM = evaluator.Confusion_Matrix()
    Acc = evaluator.get_Pixel_Accuracy()
    Acc_class = evaluator.get_Pixel_Accuracy_Class()
    Prec = evaluator.get_Precision()
    Recall = evaluator.get_Recall()
    F1_Score = evaluator.get_F1_Score(Prec, Recall)
    mIoU = evaluator.get_mIoU()
    IoU_class = evaluator.get_IoU_Class()
    # FWIoU = evaluator.get_FWIoU()

    evaluator.reset()

    if select_eval:
        print(f'    rejection ratio: {round(total_reject/total, 3)}')

    print(f'    Acc:{Acc}')
    print(f'    Acc_class:{Acc_class}')
    print(f'    Prec:{Prec}, Recall:{Recall}, F1_Score:{F1_Score}')
    print(f'    mIoU:{mIoU}')
    print(f'    IoU_class:{IoU_class}')
 
    # save_dir = args.save_dir
    # result_dir = os.path.join(save_dir, 'model_pred')
    # os.makedirs(result_dir, exist_ok = True)

    # print("Measure Performance and Save results")

    # patch_level_performance =  []
    # for b, (ids, inputs, labels, outputs, preds) in tqdm(enumerate(results), total = len(results)):
    #     for j, (id, i, l, o, p) in enumerate(zip(ids, inputs, labels, outputs, preds)): 
    #         # save_dir = os.path.join(data_dir, 'patch', parent_dir, f'{patch_mag}x_{patch_size}')

    #         # save_dir = os.path.join(result_dir, slide_name)

    #         heatmap = make_heatmap(o)
    #         overlay = cv2.addWeighted(i, 0.7, heatmap, 0.3, 0)

    #         input = Image.fromarray(np.uint8(i*255)).convert('RGB')
    #         label = Image.fromarray(np.uint8(l*255)).convert('L')
    #         overlay = Image.fromarray(np.uint8(overlay*255)).convert('RGB')
    #         pred = Image.fromarray(np.uint8(p*255)).convert('L')

    #         input.save(os.path.join(result_dir, f'{id}_input.jpg'))
    #         label.save(os.path.join(result_dir, f'{id}_label.png'))
    #         overlay.save(os.path.join(result_dir, f'{id}_heatmap_overlay.jpg'))
    #         pred.save(os.path.join(result_dir, f'{id}_pred.png'))

    #         patch_level_performance.append(get_performance(l, o, p))

    # patch_level_performance = np.nanmean(np.concatenate([patch_level_performance]), axis = 0)

    # print(f'patch-level average | accuracy: {patch_level_performance[0]:.3f} | recall: {patch_level_performance[1]:.3f} | precision: {patch_level_performance[2]:.3f} | f1 score: {patch_level_performance[3]:.3f} | AUC score: {patch_level_performance[4]:.3f}')0
    
    # save_performance_as_csv(save_dir=save_dir, performance=patch_level_performance, csv_name = "input-level_average_performance")