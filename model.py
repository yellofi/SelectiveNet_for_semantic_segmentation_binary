import torch
import torch.nn as nn


"""
UNet
"""

def CBR_2D(in_ch, out_ch, k_size = 3, stride = 1, padding = 1, bias = True):
    layers =[]
    layers += [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, bias=bias)]
    layers += [nn.BatchNorm2d(num_features=out_ch)]
    layers += [nn.ReLU()]
    conv_block = nn.Sequential(*layers)
    return conv_block

# UNet for BCE loss
class UNet_B(nn.Module):
    def __init__(self, input_type = 'RGB', selective = False):
        super(UNet_B, self).__init__()

        self.selective = selective

        if 'RGB' in input_type:
            input_ch = 3
        elif input_type == 'GH':
            input_ch = 2

        self.encoder_layer_1_1 = CBR_2D(in_ch=input_ch, out_ch=64) 
        self.encoder_layer_1_2 = CBR_2D(in_ch=64, out_ch=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder_layer_2_1 = CBR_2D(in_ch=64, out_ch=128)
        self.encoder_layer_2_2 = CBR_2D(in_ch=128, out_ch=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.encoder_layer_3_1 = CBR_2D(in_ch=128, out_ch=256)
        self.encoder_layer_3_2 = CBR_2D(in_ch=256, out_ch=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.decoder_layer_4_2 = CBR_2D(in_ch=256, out_ch=512)
        self.decoder_layer_4_1 = CBR_2D(in_ch=512, out_ch=512)

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, \
                                        kernel_size=2, stride=2, padding=0, bias=True)
        
        self.decoder_layer_3_2 = CBR_2D(in_ch=512, out_ch=256)
        self.decoder_layer_3_1 = CBR_2D(in_ch=256, out_ch=256)


        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, \
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder_layer_2_2 = CBR_2D(in_ch=256, out_ch=128)
        self.decoder_layer_2_1 = CBR_2D(in_ch=128, out_ch=128)

        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, \
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder_layer_1_2 = CBR_2D(in_ch=128, out_ch=64)
        self.decoder_layer_1_1 = CBR_2D(in_ch=64, out_ch=64)
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        if self.selective:
            self.conv_select = nn.Conv2d(64, 1, 1)
            self.conv_aux = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        enc1_1 = self.encoder_layer_1_1(x)
        enc1_2 = self.encoder_layer_1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.encoder_layer_2_1(pool1)
        enc2_2 = self.encoder_layer_2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.encoder_layer_3_1(pool2)
        enc3_2 = self.encoder_layer_3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        bottom = self.decoder_layer_4_2(pool3)
        bottom = self.decoder_layer_4_1(bottom)
        unpool3 = torch.cat((self.unpool3(bottom), enc3_2), dim=1)

        dec3_2 = self.decoder_layer_3_2(unpool3)
        dec3_1 = self.decoder_layer_3_1(dec3_2)
        unpool2 = torch.cat((self.unpool2(dec3_1), enc2_2), dim=1)

        dec2_2 = self.decoder_layer_2_2(unpool2)
        dec2_1 = self.decoder_layer_2_1(dec2_2)
        unpool1 = torch.cat((self.unpool1(dec2_1), enc1_2), dim=1)

        dec1_2 = self.decoder_layer_1_2(unpool1)
        dec1_1 = self.decoder_layer_1_1(dec1_2)

        output = self.conv1x1(dec1_1)

        if self.selective:
            select = self.conv_select(dec1_1) # (N, 1, H, W)
            aux = self.conv_aux(dec1_1) # (N, 1, H, W)
            return torch.squeeze(output, 1), torch.squeeze(select, 1), torch.squeeze(aux, 1)
        else:
            return torch.squeeze(output, 1) # (N, 1, H, W) -> (N, H, W) matching with the shape of label

# UNet for CE loss
class UNet(nn.Module):
    def __init__(self, input_type = 'RGB', n_cls = 2, selective = False):
        super(UNet, self).__init__()

        self.selective = selective

        if 'RGB' in input_type:
            input_ch = 3
        elif input_type == 'GH':
            input_ch = 2

        self.encoder_layer_1_1 = CBR_2D(in_ch=input_ch, out_ch=64) 
        self.encoder_layer_1_2 = CBR_2D(in_ch=64, out_ch=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder_layer_2_1 = CBR_2D(in_ch=64, out_ch=128)
        self.encoder_layer_2_2 = CBR_2D(in_ch=128, out_ch=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.encoder_layer_3_1 = CBR_2D(in_ch=128, out_ch=256)
        self.encoder_layer_3_2 = CBR_2D(in_ch=256, out_ch=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.decoder_layer_4_2 = CBR_2D(in_ch=256, out_ch=512)
        self.decoder_layer_4_1 = CBR_2D(in_ch=512, out_ch=512)

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, \
                                        kernel_size=2, stride=2, padding=0, bias=True)
        
        self.decoder_layer_3_2 = CBR_2D(in_ch=512, out_ch=256)
        self.decoder_layer_3_1 = CBR_2D(in_ch=256, out_ch=256)


        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, \
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder_layer_2_2 = CBR_2D(in_ch=256, out_ch=128)
        self.decoder_layer_2_1 = CBR_2D(in_ch=128, out_ch=128)

        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, \
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder_layer_1_2 = CBR_2D(in_ch=128, out_ch=64)
        self.decoder_layer_1_1 = CBR_2D(in_ch=64, out_ch=64)
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=n_cls, kernel_size=1, stride = 1)

        if self.selective:
            self.conv_select = nn.Conv2d(64, 2, 1, 1)
            self.conv_aux = nn.Conv2d(64, n_cls, 1, 1)

    def forward(self, x):
        enc1_1 = self.encoder_layer_1_1(x)
        enc1_2 = self.encoder_layer_1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.encoder_layer_2_1(pool1)
        enc2_2 = self.encoder_layer_2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.encoder_layer_3_1(pool2)
        enc3_2 = self.encoder_layer_3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        bottom = self.decoder_layer_4_2(pool3)
        bottom = self.decoder_layer_4_1(bottom)
        unpool3 = torch.cat((self.unpool3(bottom), enc3_2), dim=1)

        dec3_2 = self.decoder_layer_3_2(unpool3)
        dec3_1 = self.decoder_layer_3_1(dec3_2)
        unpool2 = torch.cat((self.unpool2(dec3_1), enc2_2), dim=1)

        dec2_2 = self.decoder_layer_2_2(unpool2)
        dec2_1 = self.decoder_layer_2_1(dec2_2)
        unpool1 = torch.cat((self.unpool1(dec2_1), enc1_2), dim=1)

        dec1_2 = self.decoder_layer_1_2(unpool1)
        dec1_1 = self.decoder_layer_1_1(dec1_2)

        output = self.conv1x1(dec1_1) # (N, C, H, W)

        if self.selective:
            select = self.conv_select(dec1_1) # (N, 2, H, W)
            aux = self.conv_aux(dec1_1) # (N, C, H, W)
            return output, select, aux
        else:
            return output

        
        


