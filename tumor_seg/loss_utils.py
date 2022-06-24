from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import numpy as np

# def make_one_hot(input, num_classes):
#     """Convert class index tensor to one hot encoding tensor.
#     Args:
#          input: A tensor of shape [N, H, W]
#          num_classes: An int of number of class
#     Returns:
#         A tensor of shape [N, num_classes, H, W]
#     """
#     n, h, w = input.shape
#     device = input.device
#     shape = (n, num_classes, h, w)
#     # shape = np.array(input.shape)
#     # shape[1] = num_classes
#     # shape = tuple(shape)
#     result = torch.zeros(shape)
#     result = result.scatter_(1, torch.unsqueeze(input, 1).cpu(), 1)
#     return result.to(device)

def calc_selective_risk(output, selection, target):

    target_coverage = 0.6
    lamb = 0.5

    if len(target.size())==1:
        target = torch.zeros(target.size(0), 2).cuda().scatter_(1, target.view(-1,1), 1)
    
    selection = F.softmax(selection, dim=1)[:,1]
    coverage = torch.mean(selection)
    zero = torch.zeros(coverage.shape).cuda()
    #print(torch.stack([target_coverage-coverage, zero], dim=-1).shape)

    loss_risk = -torch.mean(torch.sum(F.log_softmax(output, dim=1)*target, dim=1)*selection)/coverage
    diff, _ = torch.max(torch.stack([target_coverage-coverage, zero], dim=-1), dim=0)
    loss_constraint = torch.square(diff)
    print('loss_risk:', loss_risk)
    print('loss_constraint', loss_constraint)
    loss  = loss_risk + lamb*loss_constraint
    return loss, coverage

def calc_selective_risk_image(output, selection, target, target_coverage = 0.8, lamb = 8, hard_selection=False):

    # lamb = 8 # 참조 코드
    # lamb = 32 # 논문 
    # lamb = 2 

    if len(target.size())==3:
        target = torch.zeros(target.size(0), output.size(1), target.size(1), target.size(2)).cuda().scatter_(1, target.view(target.size(0),1, target.size(1), target.size(2)), 1)
    
    #target2 = 0.8*target + 0.2*(1-target)
    
    selection = F.softmax(selection, dim=1)[:,1,:,:]
    coverage = torch.mean(selection)
    zero = torch.zeros(coverage.shape).cuda()
    if hard_selection:
        selection = selection.clone().detach()
        print(selection)
        coverage = coverage.clone().detach()
        selection = torch.where(torch.tensor(selection>0.5), torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
    #print(torch.stack([target_coverage-coverage, zero], dim=-1).shape)
    
    #print(torch.sum(F.log_softmax(output, dim=1)*target, dim=1))

    # print(F.log_softmax(output, dim=1).size())
    # print(target.view(1, -1).size())
    # print((F.log_softmax(output, dim=1)*target).size())
    # print(torch.sum(F.log_softmax(output, dim=1)*target, dim=1).size())
    loss_risk = -torch.mean(torch.sum(F.log_softmax(output, dim=1)*target, dim=1)*selection)/coverage
    diff, _ = torch.max(torch.stack([target_coverage-coverage, zero], dim=-1), dim=0)
    loss_constraint = torch.square(diff)
    #loss_constraint = torch.square(1-coverage)
    
    #loss_distil = torch.mean(torch.sum(F.log_softmax(output, dim=1)*target, dim=1)*(1-selection)) - \
    #    torch.mean(torch.sum(torch.log(target2)*target2, dim=1)*(1-selection))

    #print('loss_risk:', loss_risk)
    #print('loss_constraint', loss_constraint)
    #print('coverage', coverage)
    #print('loss_distil', loss_distil)
    
    loss  = loss_risk + lamb*loss_constraint
    #loss = loss_risk + loss_distil
    return loss, coverage

if __name__ == "__main__":

    import numpy as np

    output = torch.tensor([[[[0, 1, 0], [0, 0, 1], [1, 1, 1]], [[1, 0, 1], [1, 1, 0], [0, 0, 0]]]], dtype = torch.float32).cuda()
    selection = torch.tensor([[[[0, 0, 0.1], [0, 0, 0.8], [0, 0, 0]], 
    [[1, 1, 0.9], [1, 1, 0.2], [1, 1, 1]]]], dtype = torch.float32).cuda()

    label = np.array([[[1, 0, 1], [1, 1, 1], [0, 0, 1]]])

    # label = torch.from_numpy(label)
    # label = label.type(torch.LongTensor)
    # label = label.to('cuda')

    label = np.array([[[1, 0, 1], [1, 1, 1], [0, 0, 1]]])
 
    mask = (label >= 0) & (label < 2)
    print(mask)

    output = output.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    pred = np.argmax(output, axis = -1)

    print(pred.shape)
    # print(pred)
    # print(pred.shape)
    # print(F.log_softmax(output, dim=1))
    # print(output.softmax(dim=1))
    print(2*label + pred)
    print(2*label[mask].astype('int') + pred[mask])
    print(np.bincount(2*label[mask].astype('int') + pred[mask], minlength=4))
    print(np.bincount(2*label[mask].astype('int') + pred[mask], minlength=4).reshape(2, 2))
    
    confusion_matrix = np.bincount(2*label[mask].astype('int') + pred[mask], minlength=4).reshape(2, 2)

    print(np.sum(confusion_matrix, axis = 1))

    print(np.diag(confusion_matrix)/np.sum(confusion_matrix, axis=0))


    CE_loss = torch.nn.CrossEntropyLoss()
    # print(CE_loss(output, label))

    # print(F.log_softmax(output, dim=1))
    # print(label)
    # print(F.log_softmax(output, dim=1)*label)

    # print(torch.sum(F.log_softmax(output, dim=1)*label, dim=1)) # N, H, W
    # print(torch.sum(F.log_softmax(output, dim=1)*label, dim=1).size())
    # print(-torch.sum(F.log_softmax(output, dim=1)*label)/(output.size(0)*output.size(1)*output.size(2)*output.size(3)))
    # print(CE_loss(output, label))

    # print(F.softmax(selection, dim=1)[:,1,:,:])

    # print(torch.sum(F.log_softmax(output, dim=1)*label, dim=1)*F.softmax(selection, dim=1)[:,1,:,:])


    # print(output.device, selection.device, label.device)

    # risk, coverage = calc_selective_risk_image(output, selection, label)

    # print(risk)
    # print(risk.item())

    # output = output.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    # pred = np.argmax(output, axis = -1)
    # print(pred.shape)
    # print(pred)

    # label = label.to('cpu').detach().numpy()
    # print(label.shape)
    # print(label)

    # print(np.bincount(label, minlength=4))


    