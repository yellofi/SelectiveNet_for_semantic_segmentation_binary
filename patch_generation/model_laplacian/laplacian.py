import torch
import torch.nn as nn
import torch.nn.functional as F

class Laplacian(torch.nn.Module):

    def __init__(self):
        super(Laplacian, self).__init__()

        self.conv_filter = torch.tensor([[[[0.0, 1.0, 0.0],
                                           [1.0, -4.0, 1.0],
                                            [0.0, 1.0, 0.0]]]])
        self.conv_filter = torch.FloatTensor(self.conv_filter)
        self.padding = nn.ReflectionPad2d(1)

    def forward(self, x):
        self.conv_filter = self.conv_filter.cuda()
        output = self.padding(x)
        output = F.conv2d(output, self.conv_filter, padding=0)
        return output
