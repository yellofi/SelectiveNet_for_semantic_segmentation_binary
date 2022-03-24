import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
# from torch.distributed import Backend
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
# import torch.distributed as dist
# import torch.multiprocessing as mp

from data_utils import *
from model import *

data_dir = '/mnt/hdd1/c-MET_datasets/Lung_c-MET IHC_scored/DL-based_tumor_seg_dataset'
ckpt_dir = '/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/checkpoint'
log_dir = '/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/log'

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='WSI data directory')
    parser.add_argument('--ckpt_dir', type=str, help='directory where models would be saved')
    parser.add_argument('--log_dir', type=str, help='directory where log of model training would be saved')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    print('')
    print('args={}\n'.format(args))

    return args

def train(rank, data_loader, lr, num_epoch):

    loader_train, loader_val = data_loader

    device = torch.device(f'cuda:{rank}')
    net = UNet().to(device)
    # net = DDP(net, device_ids=[rank], output_device=rank)

    optim = torch.optim.Adam(net.parameters(), lr = lr)
    fn_loss = torch.nn.BCEWithLogitsLoss().to(device)

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std : (x*std) + mean
    fn_classifier = lambda x : 1.0 * (x > 0.5)

    writer_train = SummaryWriter(log_dir = os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir = os.path.join(log_dir, 'valid'))

    start_epoch = 0
    net, optim, start_epoch = net_load(ckpt_dir = ckpt_dir, net = net, optim = optim) # 저장된 네트워크 불러오기

    for epoch in range(start_epoch+1,start_epoch+num_epoch +1):
        net.train()
        tr_loss_arr = []
        for batch, data in enumerate(loader_train, 1):
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

            writer_train.add_image('label', label, len(loader_train) * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', inputs, len(loader_train) * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, len(loader_train) * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(tr_loss_arr), epoch)

        # validation
        with torch.no_grad(): # validation 이기 때문에 backpropa 진행 x, 학습된 네트워크가 정답과 얼마나 가까운지 loss만 계산
            net.eval() # 네트워크를 evaluation 용으로 선언
            val_loss_arr = []

            for batch, data in enumerate(loader_val,1):
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

                writer_val.add_image('label', label, len(loader_val) * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', inputs, len(loader_val) * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, len(loader_val) * (epoch - 1) + batch, dataformats='NHWC')

            writer_val.add_scalar('loss', np.mean(val_loss_arr), epoch)
            # print('valid : epoch %04d / %04d | Loss %.05f'%(epoch, num_epoch, np.mean(val_loss_arr)))

            writer_train.close()
            writer_val.close()

            print('epoch %04d / %04d | train_loss %.05f | valid_loss %.05f'%(epoch, num_epoch, np.mean(tr_loss_arr), np.mean(val_loss_arr)))

        net_save(ckpt_dir=ckpt_dir, net = net, optim = optim, epoch = epoch)

def main(rank, world_size):

    lr = 1e-3
    batch_size = 16
    num_epoch = 100

    print(f'# of gpu: {world_size}, gpu id: {rank}\n')

    data_loader = create_data_loader(data_dir=data_dir, batch_size=batch_size)
    train(rank=rank, data_loader=data_loader, lr=lr, num_epoch=num_epoch)


if __name__ == '__main__':

    # args = parse_arguments()

    rank = 3
    world_size = torch.cuda.device_count() #8

    torch.cuda.set_device(rank)
    
    # os.environ['MASTER_ADDR'] = '192.168.0.38'
    # os.environ['MASTER_PORT'] = '10011'

    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    main(rank=rank, world_size=world_size)
    
    # mp.spawn(main, 
    #         args=(world_size,),
    #         nprocs=world_size,
    #         join=True)
    
    