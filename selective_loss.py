from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import numpy as np

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, H, W]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, H, W]
    """
    n, h, w = input.shape
    device = input.device
    shape = (n, num_classes, h, w)
    # shape = np.array(input.shape)
    # shape[1] = num_classes
    # shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, torch.unsqueeze(input, 1).cpu(), 1)
    return result.to(device)

def calc_selective_risk_image(output, selection, target, target_coverage = 0.8, lamb = 8, hard_selection=False):
    """
    the modificated selective risk for image segmentation with Cross Entropy Loss (N classes)

    Args
        output: (N, C, H, W)
        selection: (N, C, H, W)
        target: (N, H, W) or (N, C, H, W)
    Return 
        selective loss
    """
    # lamb = 32 # lambda at the original paper

    # make one_hot
    if len(target.size())==3:
        target = torch.zeros(target.size(0), output.size(1), target.size(1), target.size(2)).cuda().scatter_(1, target.view(target.size(0),1, target.size(1), target.size(2)), 1)
    
    selection = F.softmax(selection, dim=1)[:,1,:,:]
    coverage = torch.mean(selection)
    zero = torch.zeros(coverage.shape).cuda()
    if hard_selection:
        selection = selection.clone().detach()
        print(selection)
        coverage = coverage.clone().detach()
        selection = torch.where(torch.tensor(selection>0.5), torch.tensor(1.).cuda(), torch.tensor(0.).cuda())

    loss_risk = -torch.mean(torch.sum(F.log_softmax(output, dim=1)*target, dim=1)*selection)/coverage
    diff, _ = torch.max(torch.stack([target_coverage-coverage, zero], dim=-1), dim=0)
    loss_constraint = torch.square(diff)
    
    loss  = loss_risk + lamb*loss_constraint

    return loss, coverage

def calc_selective_risk_image_b(output, selection, target, target_coverage = 0.8, lamb = 8, hard_selection=False):
    """
    the modificated selective risk for image segmentation with BCEwithLogitLoss (Binary Class)

    Args
        output: (N, H, W)
        selection: (N, H, W)
        target: (N, H, W)
    Return 
        selective loss
    """
   # lamb = 32 # lambda at the original paper
    
    selection = torch.sigmoid(selection)
    coverage = torch.mean(selection)
    zero = torch.zeros(coverage.shape).cuda()
    if hard_selection:
        selection = selection.clone().detach()
        coverage = coverage.clone().detach()
        selection = torch.where(torch.tensor(selection>0.5), torch.tensor(1.).cuda(), torch.tensor(0.).cuda())

    prob = torch.sigmoid(output)
    loss_risk = -torch.mean((target*torch.log(prob)+(1-target)*torch.log(1-prob))*selection)/coverage 
    diff, _ = torch.max(torch.stack([target_coverage-coverage, zero], dim=-1), dim=0)
    loss_constraint = torch.square(diff)
    
    loss  = loss_risk + lamb*loss_constraint
    return loss, coverage


    