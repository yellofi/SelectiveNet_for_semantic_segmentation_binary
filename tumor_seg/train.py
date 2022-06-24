import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
# from torch.distributed import Backend
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
# import torch.distributed as dist
# import torch.multiprocessing as mp
from tqdm import tqdm

from data_utils import *
from Dataset_sample import Dataset
from Dataset_samsung import *
from model import *
from net_utils import *
from loss_utils import *
from compute_metric import Evaluator

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='WSI data directory',
                        default = '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset/2205_1차anno')
    parser.add_argument('--model_dir', type=str, help='directory where logs and models would be saved',
                        default = '/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/06_baseline_samsung_data/dp')
    
    parser.add_argument('--input_type', type=str, default='RGB')
    parser.add_argument('--patch_mag', type=int, default=200)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--n_cls', type=int, default=2)

    parser.add_argument('--local_rank', type=int, nargs='+', default=[0], help='local rank')
    parser.add_argument('--fold', type = int, default = 1, help = 'which fold in 5-fold cv')
    parser.add_argument('--data_type', type=str, default = 'samsung', 
                        help='samsung have no each fold directory (sample or samsung)')

    parser.add_argument('--model_arch', type=str, default = 'UNet', choices = ['UNet', 'UNet_B'])
    parser.add_argument('--selective', type=bool, default = False)
    parser.add_argument('--s_lamb', type=int, default = 32)
    
    parser.add_argument('--optim', type=str, default='Adam', choices = ['Adam', 'SGD'])
    parser.add_argument('--momentum', type=float, default=0, choices = [0.9])
    parser.add_argument('--w_decay', type=float, default=0, choices = [5e-4])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_sche', type=str, default = None, choices = ['StepLR', 'ReduceLR', 'CosineAnnealingLR'])
    parser.add_argument('--patience', type=int, default = 10)
    parser.add_argument('--factor', type=float, default = 0.5)
    parser.add_argument('--lr_min', type=float, default = 1e-5)

    parser.add_argument('--loss', type=str, default = 'CE', choices = ['BCElogit', 'CE'])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=100)
    
    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

def train(args, data_loader, ckpt_dir, log_dir):

    rank = args.local_rank
    
    input_type = args.input_type
    n_cls = args.n_cls

    num_epoch = args.n_epoch
    lr = args.lr
  
    loader_train, loader_val = data_loader
    start_epoch = 0

    if args.model_arch == 'UNet':
        net = UNet(input_type, n_cls, selective=args.selective)
    elif args.model_arch == 'UNet_B':
        net = UNet_B(input_type) # for BCE loss, output (N, 1, H, W)

    if args.optim == 'Adam':
        optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.w_decay)
    elif args.optim == 'SGD':
        optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.w_decay) 
        # SelectiveNet 논문 momentum 0.9 weight decay 5e-4 (clssification)

    if args.lr_sche == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.patience, gamma=args.factor)
    elif args.lr_sche == 'ReduceLR':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode ='min', patience=args.patience, factor = args.factor)
    elif args.lr_sche == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = args.patience, eta_min = args.lr_min)

    if len(rank) != 1: # gpu 여러개 쓰고 싶을때, DP 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif len(rank) == 1: # 0번 말고 다른 gpu도 single로 쓰고 싶을때
        device = torch.device(f'cuda:{rank}')
        net = net.to(device)

    # load pre-trained weights 
    if not os.path.exists(ckpt_dir): # 저장된 네트워크가 없다면 인풋을 그대로 반환
        start_epoch = 0
    else: 
        ckpt_lst = os.listdir(ckpt_dir) # ckpt_dir 아래 있는 모든 파일 리스트를 받아온다
        ckpt_lst.sort(key = lambda f : int(''.join(filter(str.isdigit,f))))

        if len(rank) != 1:
            ckpt = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[-1]), map_location='cpu')
        elif len(rank) == 1:
            ckpt = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[-1]), map_location=device)

        try: ckpt['net'] = remove_module(ckpt)
        except: pass

        net.load_state_dict(ckpt['net'])
        # optim.load_state_dict(ckpt['optim'])
        start_epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

        print('Load weights from', os.path.join(ckpt_dir, ckpt_lst[-1]))

    if len(rank) > 1:
        net = torch.nn.DataParallel(net, device_ids=rank)
        net = net.cuda()
  
    if args.loss == 'BCElogit':
        loss_A = torch.nn.BCEWithLogitsLoss() # (N, D) 
    elif args.loss == 'CE':
        loss_A = torch.nn.CrossEntropyLoss() # (N, C, D) one hot encoding is required

    if args.selective:
        loss_S = calc_selective_risk_image

    fn_denorm = lambda x, mean, std : (x*std) + mean # input -> [0, 1]로 다시 복원하기 위해
    fn_classifier = lambda x : 1.0 * (x > 0.5)

    writer_train = SummaryWriter(log_dir = os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir = os.path.join(log_dir, 'valid'))

    evaluator = Evaluator(num_class=n_cls, selective=args.selective)

    for epoch in range(start_epoch+1,start_epoch+num_epoch +1):

        for param_group in optim.param_groups:
            current_lr = (param_group['lr'])

        writer_train.add_scalar('lr', current_lr, epoch)

        print(f'epoch {epoch} / {start_epoch+num_epoch}, learning rate {current_lr}')
        net.train()

        tr_loss_arr = []
        val_loss_arr = []

        if args.selective:
            tr_total, val_total = 0, 0
            tr_total_reject, val_total_reject = 0, 0
            tr_aux_loss_arr, tr_sel_loss_arr = [], []
            val_aux_loss_arr, val_sel_loss_arr = [], []

        for data in tqdm(loader_train, total = len(loader_train), desc = 'train'):
        
            # forward
            inputs = data['input'].to(device) # (N, C, H, W)
            label = data['label'] # (N, H, W), torch.int64

            if 'BCE' in args.loss:
                label = label.type(torch.FloatTensor) # torch.flaot32 for BCE
            label = label.to(device) # (N, H, W)

            if args.selective:
                output, selection, aux  = net(inputs)
                aux_loss = loss_A(aux, label)
                select_loss, coverage = loss_S(output, selection, target=label, lamb=args.s_lamb)

                tr_aux_loss_arr += [aux_loss.item()]
                tr_sel_loss_arr += [select_loss.item()]
                
                loss = aux_loss + select_loss
                
            else:
                output = net(inputs)
                loss = loss_A(output, label)

            # backward
            optim.zero_grad()  # gradient 초기화
            loss.backward() # gradient backpropagation
            optim.step() # backpropa 된 gradient를 이용해서 각 layer의 parameters update
            
            # inputs = fn_tonumpy(fn_denorm(inputs,0.5,0.5)) # (N, C, H, W) ~ [mean: 0.5, std: 0.5] -> (N, H, W, C) ~ [0, 1]
            label = label.to('cpu').detach().numpy()

            if len(output.size()) == 4:
                output = output.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
                pred = np.argmax(output, axis=-1) # (N, H, W, C) -> (N, H, W), [0.2, 0.7, 0.1] -> [1] 
            elif len(output.size()) == 3:
                output = output.to('cpu').detach().numpy()
                pred = fn_classifier(output) # binary class 0.4 -> 0, 0.6 -> 1

            label = label.astype('uint8')
            pred = pred.astype('uint8')
     
            if args.selective:
                _, selection = torch.max(selection, dim=1)
                evaluator.add_batch(label, pred, selection=selection.to('cpu').detach().numpy())
                tr_total += label.size
                reject = label.size - selection.sum().item()
                tr_total_reject += reject
            else:
                evaluator.add_batch(label, pred)

            tr_loss_arr += [loss.item()]

        tr_acc = evaluator.get_Pixel_Accuracy()
        evaluator.reset()
        
        if args.lr_sche != None:
            if args.lr_sche == 'ReduceLR':
                scheduler.step(np.mean(tr_loss_arr))
            else:
                scheduler.step()

        # optim.zero_grad()
        # torch.cuda.empty_cache()

        writer_train.add_scalar('loss', np.mean(tr_loss_arr), epoch)
        writer_train.add_scalar('accuracy', tr_acc, epoch)

        if args.selective:
            writer_train.add_scalar('aux loss', np.mean(tr_aux_loss_arr), epoch)
            writer_train.add_scalar('selection loss', np.mean(tr_sel_loss_arr), epoch)
            writer_train.add_scalar('rejection ratio', tr_total_reject/tr_total, epoch)

        # validation
        with torch.no_grad(): 
            net.eval()

            for data in tqdm(loader_val, total = len(loader_val), desc = 'valid'):
                # forward
                inputs = data['input'].to(device) # (N, C, H, W)
                label = data['label'] # (N, H, W), torch.int64

                if 'BCE' in args.loss:
                    label = label.type(torch.FloatTensor) # torch.flaot32
                label = label.to(device) # (N, H, W)
                
                if args.selective:
                    output, selection, aux  = net(inputs)
                    aux_loss = loss_A(aux, label)
                    select_loss, coverage = loss_S(output, selection, label, lamb=args.s_lamb)

                    val_aux_loss_arr += [aux_loss.item()]
                    val_sel_loss_arr += [select_loss.item()]

                    loss = aux_loss + select_loss
                else:
                    output = net(inputs)
                    loss = loss_A(output, label)

                # inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
                label = label.to('cpu').detach().numpy()

                if len(output.size()) == 4:
                        output = output.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
                        pred = np.argmax(output, axis=-1) # (N, H, W, C) -> (N, H, W), [0.2, 0.7, 0.1] -> [1] 
                elif len(output.size()) == 3:
                    output = output.to('cpu').detach().numpy()
                    pred = fn_classifier(output) # binary class 0.4 -> 0, 0.6 -> 1

                label = label.astype('uint8')
                pred = pred.astype('uint8')

                if args.selective:
                    _, selection = torch.max(selection, dim=1)
                    evaluator.add_batch(label, pred, selection=selection.to('cpu').detach().numpy())
                    val_total += label.size
                    reject = label.size - selection.sum().item()
                    val_total_reject += reject
                else:
                    evaluator.add_batch(label, pred)

                val_loss_arr += [loss.item()]

        val_acc = evaluator.get_Pixel_Accuracy()
        evaluator.reset()

        writer_val.add_scalar('loss', np.mean(val_loss_arr), epoch)
        writer_val.add_scalar('accuracy', val_acc, epoch)

        if args.selective:
            writer_val.add_scalar('aux loss', np.mean(val_aux_loss_arr), epoch)
            writer_val.add_scalar('selection loss', np.mean(val_sel_loss_arr), epoch)
            writer_val.add_scalar('rejection ratio', val_total_reject/val_total, epoch)

        writer_train.close()
        writer_val.close()

        print('train_loss %.05f train_acc %.04f | valid_loss %.05f valid_acc %.04f'%(np.mean(tr_loss_arr), tr_acc, np.mean(val_loss_arr), val_acc))

        if args.selective:
            print('train_aux_loss %.05f | train_select_loss %.05f | train_rejection %.03f'%(np.mean(tr_aux_loss_arr), np.mean(tr_sel_loss_arr), tr_total_reject/tr_total))
            print('valid_aux_loss %.05f | valid_select_loss %.05f | valid_rejection %.03f'%(np.mean(val_aux_loss_arr), np.mean(val_sel_loss_arr), val_total_reject/val_total))

        net_save(ckpt_dir=ckpt_dir, net = net, optim = optim, epoch = epoch)

# def main(rank, world_size):

#     print(f'# of gpu: {world_size}, gpu id: {rank}\n')
#     data_loader = create_data_loader(data_dir, batch_size, input_type)
#     train(rank, data_loader, lr, num_epoch, input_type)

if __name__ == '__main__':

    args = parse_arguments()
    
    rank = args.local_rank
    world_size = torch.cuda.device_count() #8

    # torch.cuda.set_device(rank[0])

    # os.environ['MASTER_ADDR'] = '192.168.0.38'
    # os.environ['MASTER_PORT'] = '10011'

    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    transform_train = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    transform_val = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    input_type = args.input_type
    patch_mag = args.patch_mag
    patch_size = args.patch_size
    batch_size = args.batch_size

    if args.data_type == 'sample':
        data_dir = f'{args.data_dir}/{args.fold}-fold'
        dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'), input_type=input_type, transform=transform_train)
        dataset_val = Dataset(data_dir=os.path.join(data_dir, 'valid'), input_type=input_type, transform=transform_val) 
    elif args.data_type == 'samsung':
        data_dir = args.data_dir
        train_list, valid_list = construct_train_valid(data_dir, test_fold = args.fold)
        dataset_train = SamsungDataset(data_dir, train_list, patch_mag, patch_size, input_type, transform=transform_train)
        dataset_val = SamsungDataset(data_dir, valid_list, patch_mag, patch_size, input_type, transform=transform_val)

    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True, num_workers=16)
    loader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle=False)

    print(f'# of gpu: {world_size}, gpu id: {rank}\n')

    ckpt_dir = f'{args.model_dir}/{args.fold}-fold/checkpoint'
    log_dir = f'{args.model_dir}/{args.fold}-fold/log'

    train(args, (loader_train, loader_val), ckpt_dir, log_dir)

    # main(rank=rank, world_size=world_size)
    
    # mp.spawn(main, 
    #         args=(world_size,),
    #         nprocs=world_size,
    #         join=True)