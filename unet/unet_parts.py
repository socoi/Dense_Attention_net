# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .deformable_convs import DeformConv2d


class double_conv_cam(nn.Module):
    def __init__(self, in_ch, out_ch, set_dilation=1):
        super(double_conv_cam, self).__init__()
        self.l0 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        #DeformConv2d(in_ch, out_ch, modulation=True),
        self.l1 = nn.BatchNorm2d(out_ch)
        self.l2 = nn.ReLU(inplace=True)
        self.l3 = nn.Dropout(0.2)
        self.l4 = nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=set_dilation)
        #DeformConv2d(out_ch, out_ch, modulation=True),
        self.l5 = nn.BatchNorm2d(out_ch)
        self.l6 = nn.ReLU(inplace=True)

        # l4 weight
        self.weight = list(self.l4.parameters())[-2]
        self.out_ch = out_ch

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        # x_2 cam
        weight = torch.squeeze(self.weight).cuda()  # (2, channel)
        print(self.weight)
        print(weight.shape)
        batch_weight = torch.empty((x.shape[0], 1, self.out_ch)).cuda() # bz, 1, channel
        batch_weight[:, :, :] = weight[1, :] #假定这里1是vessel权重

        bz, nc, h, w = x.shape
        feature_batch = x.reshape((bz, nc, h * w))
        cam = torch.einsum('ijk,ikl->ijl', [batch_weight, feature_batch])
        cam = torch.reshape(cam, (x.shape[0], 1, 48, 48))
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        del batch_weight, weight
        x = self.l5(x)
        x = self.l6(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            #DeformConv2d(in_ch, out_ch, modulation=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #DeformConv2d(out_ch, out_ch, modulation=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# class double_conv(nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#     def __init__(self, in_ch, out_ch):
#         super(double_conv, self).__init__()
#         self.startconv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.LeakyReLU(inplace=True)
#         )
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             #DeformConv2d(in_ch, out_ch, modulation=True),
#             nn.BatchNorm2d(out_ch),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             #DeformConv2d(out_ch, out_ch, modulation=True),
#             nn.BatchNorm2d(out_ch),
#         )
#         self.LeakyReLU = nn.LeakyReLU(inplace=True)


#     def forward(self, x):
#         x = self.startconv(x)
#         identity = x
#         out = self.conv(x)
#         out += identity
#         out = self.LeakyReLU(out)
#         return out


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        #self.conv = double_conv_cam(in_ch, out_ch, set_dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
            #double_conv_cam(in_ch, out_ch, set_dilation=dilation)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class attention_gate(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(attention_gate, self).__init__()
        self.kernel_size = kernel_size
        # self.low_level = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        # )
        self.high_level = nn.Sequential(
            nn.AvgPool2d(self.kernel_size),
            nn.Conv2d(out_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x2, x1): # x1: high-level features, need upsample
        x_1 = self.high_level(x1)
        #x_2 = self.low_level(x2)
        x_2 = x2
        b1, c1, w1, h1 = x_1.shape
        b2, c2, w2, h2 = x_2.shape
        a = x_1.view(b1 * c1, w1, h1)
        b = x_2.view(b2 * c2, w2, h2)
        a = a.repeat(1, w1, h1)
        c = a * b
        c = c.view(b2, c2, w2, h2)
        #x1 = self.up(x1)
        #x = torch.cat([c, x1], dim=1)
        #x = self.conv(x)
        #print(x.shape)
        return c


# class attention_up(nn.Module):
#     # single_up: dont need double_conv later
#     # nested U-net just transconv

#     def __init__(self, in_ch, out_ch, bilinear=True, kernel_size=1):
#         super(attention_up, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(
#                 scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
#         self.attention = attention_gate(in_ch, out_ch, kernel_size)
#         #self.conv = double_conv(in_ch + out_ch, out_ch) #upsamplecase
#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x1, x2): # x1: high-level features
#         c = self.attention(x1, x2)
#         x1 = self.up(x1)
#         x = torch.cat([c, x1], dim=1)
#         x = self.conv(x)
#         #print(x.shape)
#         return x

# single_up: dont need double_conv later
# nested U-net just transconv
class up(nn.Module):
    def __init__(self, in_ch, out_ch, single_up=False, bilinear=True, GCN_module=False, attention_up=False, kernel_size=0, dilation=1):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            #self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.up = nn.ConvTranspose2d(in_ch, out_ch - in_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.singleup = single_up
        self.attention_up = attention_gate(out_ch - in_ch, in_ch, kernel_size)
        self.reduce_dim = nn.Sequential(
        nn.Conv2d(in_ch, out_ch - in_ch, 3, padding=1),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        #self.weight = list(self.conv.parameters())[-2]

        self.GCN_module = GCN_module
        if GCN_module:
            #self.GCN = _GlobalConvModule(in_ch, out_ch, (7, 7))
            #self.BR = _BoundaryRefineModule(out_ch)
            self.GCN = _GlobalConvModule(in_ch // 2, in_ch // 2, (7, 7))
            self.BR = _BoundaryRefineModule(in_ch // 2)

    def forward(self, x1, x2): #x1 need upsample
        if self.GCN_module:
            x1 = self.GCN(x1)
            x1 = self.BR(x1)
        if self.attention_up:
            x2 = self.attention_up(x2, x1)
        #x1 = self.up(x1)
        x1 = self.reduce_dim(x1)
        x = torch.cat([x2, x1], dim=1)
        if not self.singleup:
            x = self.conv(x)
        return x



class up1(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, GCN_module=False, attention_up=False, kernel_size=0):
        super(up1, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.use_attention = attention_up
        self.attention_up = attention_gate(out_ch - in_ch, in_ch, kernel_size)
        self.reduce_dim = nn.Sequential(
        nn.Conv2d(in_ch, 2 * in_ch - out_ch, 3, padding=1),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch - in_ch, 2, stride=2)

    def forward(self, x3, x2, x1): # last need upsample
        # input is CHW
        if self.use_attention:
            x2 = torch.cat([x2, x3], dim=1)
            x = self.attention_up(x2, x1)
        x1 = self.reduce_dim(x1)
        x = torch.cat([x1, x], dim=1)
        x = self.conv(x)
        return x

class up2(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, GCN_module=False, attention_up=False, kernel_size=0):
        super(up2, self).__init__()
        self.GCN_module = GCN_module
        if attention_up:
            self.use_attention = attention_up
            self.attention_up = attention_gate(out_ch - in_ch, in_ch, kernel_size)
        if GCN_module:
            self.GCN = _GlobalConvModule(in_ch, out_ch, (7, 7))
            self.BR = _BoundaryRefineModule(out_ch)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch - in_ch, 2, stride=2)

    def forward(self, x4, x3, x2, x1):
        if self.GCN_module:
            x4 = self.GCN(x4)
            x4 = self.BR(x4)
        if self.use_attention:
            x4 = self.attention_up(x4, x1)
            x3 = self.attention_up(x3, x1)
            x2 = self.attention_up(x2, x1)
        x1 = self.up(x1)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class up3(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, GCN_module=False, attention_up=False, kernel_size=0):
        super(up3, self).__init__()
        self.GCN_module = GCN_module
        if attention_up:
            self.attention_up = attention_gate(out_ch - in_ch, in_ch, kernel_size)
            self.use_attention = attention_up
        if GCN_module:
            self.GCN = _GlobalConvModule(in_ch, out_ch, (7, 7))
            self.BR = _BoundaryRefineModule(out_ch)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch - in_ch, 2, stride=2)

    def forward(self, x5, x4, x3, x2, x1):
        if self.GCN_module:
            x5 = self.GCN(x5)
            x5 = self.BR(x5)
        if self.use_attention:
            x5 = self.attention_up(x5, x1)
            x4 = self.attention_up(x4, x1)
            x3 = self.attention_up(x3, x1)
            x2 = self.attention_up(x2, x1)
        x1 = self.up(x1)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, CAM_outout=False):
        super(outconv, self).__init__()
        # self.conv = nn.Sequential(
        # nn.Conv2d(in_ch, out_ch, 1),
        # nn.BatchNorm2d(out_ch),
        # nn.ReLU(inplace=True)
        # )
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.CAM_outout = CAM_outout
        if CAM_outout:
            self.in_ch = in_ch
            self.weight = list(self.conv.parameters())[-2]

    def forward(self, x):
        if self.CAM_outout: #only consider 0_4 for cam 32*34
            weight = torch.squeeze(self.weight).cuda()  # (2, channel)
            batch_weight = torch.empty((x.shape[0], 1, 32 * 34)).cuda()
            batch_weight[:, :, :] = weight[1, :] #假定这里1是vessel权重

            bz, nc, h, w = x.shape
            feature_batch = x.reshape((bz, nc, h * w))
            cam = torch.einsum('ijk,ikl->ijl', [batch_weight, feature_batch])
            cam = torch.reshape(cam, (x.shape[0], 1, 48, 48))
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)
            del batch_weight, weight
            return cam
        else:
            x = self.conv(x)
            return x

class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x