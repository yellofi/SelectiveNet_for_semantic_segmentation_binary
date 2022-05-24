import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
# from torch.distributed import Backend
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
# import torch.distributed as dist
# import torch.multiprocessing as mp
from tqdm import tqdm

from data_utils import *
from model import *

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='WSI data directory',
                        default = '/mnt/hdd1/c-MET_datasets/SLIDE_DATA/DL-based_tumor_seg_dataset/2205_1차anno')
    parser.add_argument('--model_dir', type=str, help='directory where logs and models would be saved',
                        default = '/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/06_baseline_samsung_data/dp')

    parser.add_argument('--local_rank', type=int, nargs='+', default=[0], help='local rank')
    parser.add_argument('--fold', type = int, default = 1, help = 'which fold in 5-fold cv')
    parser.add_argument('--data_type', type=str, default = 'samsung', 
                        help='samsung have no each fold directory (sample or samsung)')

    parser.add_argument('--input_type', type=str, default='RGB')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=100)
    
    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

def create_data_loader(data_dir: str, batch_size: int, input_type: str) -> Tuple[DataLoader, DataLoader]: 

    # transform_train = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), PartialNonTissue(), ToTensor()])
    transform_train = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'),transform = transform_train, input_type = input_type)
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)

    transform_val = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'valid'), transform = transform_val, input_type = input_type)
    loader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle=True)

    return loader_train, loader_val

def train(rank, data_loader, lr, num_epoch, input_type):

    loader_train, loader_val = data_loader

    # net = DDP(net, device_ids=[rank], output_device=rank)
    start_epoch = 0

    if len(rank) != 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = UNet(input_type, DataParallel=True)
        optim = torch.optim.Adam(net.to(device).parameters(), lr = lr)
        net, optim, start_epoch = net_load(ckpt_dir=ckpt_dir, net=net, optim=optim)
        net = torch.nn.DataParallel(net, device_ids=rank)
        net = net.to(device)

    else:
        # single gpu -> device map location으로 불러와야 gpu 0을 안 씀
        device = torch.device(f'cuda:{rank}')
        net = UNet(input_type).to(device)
        optim = torch.optim.Adam(net.parameters(), lr = lr)
        net, optim, start_epoch = net_load(ckpt_dir=ckpt_dir, net=net, optim=optim, device=device) 
  
    fn_loss = torch.nn.BCEWithLogitsLoss().to(device)

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean
    fn_classifier = lambda x : 1.0 * (x > 0.5)

    writer_train = SummaryWriter(log_dir = os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir = os.path.join(log_dir, 'valid'))

    for epoch in range(start_epoch+1,start_epoch+num_epoch +1):
        net.train()
        tr_loss_arr = []

        # for batch, data in enumerate(loader_train, 1):
        for batch, data in enumerate(tqdm(loader_train, total = len(loader_train), postfix = 'train'), 1):
        
            # forward
            # label = data['label'].to(device, non_blocking=True)   
            # inputs = data['input'].to(device, non_blocking=True)
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            
            output = net(inputs) 

            # backward
            optim.zero_grad()  # gradient 초기화
            loss = fn_loss(output, label)  # output과 label 사이의 loss 계산
            loss.backward() # gradient backpropagation
            optim.step() # backpropa 된 gradient를 이용해서 각 layer의 parameters update

            # save loss
            tr_loss_arr += [loss.item()]

            # tensorbord에 결과값들 저장하기
            label = fn_tonumpy(label)
            inputs = fn_tonumpy(fn_denorm(inputs,0.5,0.5))
            output = fn_tonumpy(fn_classifier(output))

            # writer_train.add_image('label', label, len(loader_train) * (epoch - 1) + batch, dataformats='NHWC')
            # if input_type == 'RGB':
                # writer_train.add_image('input', inputs, len(loader_train) * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('output', output, len(loader_train) * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(tr_loss_arr), epoch)

        # validation
        with torch.no_grad(): # validation 이기 때문에 backpropa 진행 x, 학습된 네트워크가 정답과 얼마나 가까운지 loss만 계산
            net.eval() # 네트워크를 evaluation 용으로 선언
            val_loss_arr = []

            # for batch, data in enumerate(loader_val,1):
            for batch, data in enumerate(tqdm(loader_val, total = len(loader_val), postfix = 'valid'), 1):
                # # forward
                # label = data['label'].to(device, non_blocking=True)
                # inputs = data['input'].to(device, non_blocking=True)
                label = data['label'].to(device)
                inputs = data['input'].to(device)
                output = net(inputs)

                # loss 
                loss = fn_loss(output,label)
                val_loss_arr += [loss.item()]
                # print('valid : epoch %04d / %04d | Batch %04d \ %04d | Loss %.05f'%(epoch,num_epoch,batch,len(loader_val),np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_classifier(output))

                # writer_val.add_image('label', label, len(loader_val) * (epoch - 1) + batch, dataformats='NHWC')
                # if input_type == 'RGB':
                    # writer_val.add_image('input', inputs, len(loader_val) * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('output', output, len(loader_val) * (epoch - 1) + batch, dataformats='NHWC')

            writer_val.add_scalar('loss', np.mean(val_loss_arr), epoch)
            # print('valid : epoch %04d / %04d | Loss %.05f'%(epoch, num_epoch, np.mean(val_loss_arr)))

            writer_train.close()
            writer_val.close()

            print('epoch %04d / %04d | train_loss %.05f | valid_loss %.05f'%(epoch, start_epoch+num_epoch, np.mean(tr_loss_arr), np.mean(val_loss_arr)))

        net_save(ckpt_dir=ckpt_dir, net = net, optim = optim, epoch = epoch)

# def main(rank, world_size):

#     print(f'# of gpu: {world_size}, gpu id: {rank}\n')

#     data_loader = create_data_loader(data_dir, batch_size, input_type)

#     train(rank, data_loader, lr, num_epoch, input_type)


if __name__ == '__main__':

    args = parse_arguments()
    
    rank = args.local_rank
    world_size = torch.cuda.device_count() #8


    torch.cuda.set_device(rank[0])

    # os.environ['MASTER_ADDR'] = '192.168.0.38'
    # os.environ['MASTER_PORT'] = '10011'

    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    transform_train = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    transform_val = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    input_type = args.input_type
    batch_size = args.batch_size

    if args.data_type == 'sample':
        data_dir = f'{args.data_dir}/{args.fold}-fold'
        dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'),transform = transform_train, input_type = input_type)
        dataset_val = Dataset(data_dir=os.path.join(data_dir, 'valid'), transform = transform_val, input_type = input_type) 

    elif args.data_type == 'samsung':
        data_dir = args.data_dir
        train_list, valid_list = construct_train_valid(data_dir, test_fold = args.fold)

        dataset_train = SamsungDataset(data_dir = data_dir, data_list = train_list, transform = transform_train, input_type = input_type)
        dataset_val = SamsungDataset(data_dir= data_dir, data_list = valid_list, transform = transform_val, input_type = input_type) 
    
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle=True)

    print(f'# of gpu: {world_size}, gpu id: {rank}\n')

    lr = args.lr
    
    num_epoch = args.n_epoch

    ckpt_dir = f'{args.model_dir}/{args.fold}-fold/checkpoint'
    log_dir = f'{args.model_dir}/{args.fold}-fold/log'

    train(rank, (loader_train, loader_val), lr, num_epoch, input_type)

    # main(rank=rank, world_size=world_size)
    
    # mp.spawn(main, 
    #         args=(world_size,),
    #         nprocs=world_size,
    #         join=True)
    
    