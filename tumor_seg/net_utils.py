import os
from typing import OrderedDict
import torch

# 네트워크 저장하기
# train을 마친 네트워크 저장 
# net : 네트워크 파라미터, optim  두개를 dict 형태로 저장
def net_save(ckpt_dir,net,optim,epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},'%s/model_epoch%d.pth'%(ckpt_dir,epoch))

# DP로 저장된 model은 dict의 key앞에 module이 붙음 이걸 제거해줌
def remove_module(ckpt):
    net_state_dict = OrderedDict()
    for k, v in ckpt['net'].items():
        name = k.replace("module.", "")
        net_state_dict[name] = v
    return net_state_dict

# 마지막 epoch의 모델의 weight을 가져옴 
# net, optim, epoch
def net_train_load(ckpt_dir, net, optim, device=None):
    if not os.path.exists(ckpt_dir): # 저장된 네트워크가 없다면 인풋을 그대로 반환
        epoch = 0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir) # ckpt_dir 아래 있는 모든 파일 리스트를 받아온다
    ckpt_lst.sort(key = lambda f : int(''.join(filter(str.isdigit,f))))

    print('model: ', ckpt_lst[-1])

    if device != None:
        ckpt = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[-1]), map_location=device)
    else:
        ckpt = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[-1]), map_location='cpu')

    try: ckpt['net'] = remove_module(ckpt)
    except: pass

    net.load_state_dict(ckpt['net'])
    optim.load_state_dict(ckpt['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net,optim,epoch

# model_path에 있는 epoch의 모델의 weight만 가져옴 
def net_test_load(model_path, net, device=None):
    if device != None:
        ckpt = torch.load(model_path, map_location=device)
    else:
        ckpt = torch.load(model_path)

    try: ckpt['net'] = remove_module(ckpt)
    except: pass
    net.load_state_dict(ckpt['net'])

    # print('     model: ', model_path)
    return net