# full assembly of the sub-parts to form the complete net
import torch.nn.functional as F
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = attention_up(512, 256, bilinear=False, kernel_size=3)
        self.up2 = attention_up(256, 128, bilinear=False, kernel_size=6)
        self.up3 = attention_up(128, 64, bilinear=False, kernel_size=12)
        self.up4 = attention_up(64, 32, bilinear=False, kernel_size=24)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.softmax(x, dim=1)
        #return F.sigmoid(x)


class Nested_unet(nn.Module):
    def __init__(self, n_channels, n_classes, init_channel=32, patch_size=48):
        super(Nested_unet, self).__init__()
        self.init_channel = init_channel
        self.inc = inconv(n_channels, init_channel)
        self.down1 = down(init_channel, init_channel * 2)
        self.down2 = down(init_channel * 2, init_channel * 4)
        self.down3 = down(init_channel * 4, init_channel * 8)
        self.down4 = down(init_channel * 8, init_channel * 16)

        #Unet++,
        self.up0_1 = up(init_channel * 2, init_channel * 3, attention_up=True, kernel_size=patch_size // 2) # 0.1 -> 128 + 64 = 192
        self.up1_1 = up(init_channel * 4, init_channel * 6, attention_up=True, kernel_size=patch_size // 4) # 1.1 -> 256 + 128 = 384
        self.up2_1 = up(init_channel * 8, init_channel * 12, attention_up=True, kernel_size=patch_size // 8) # 2.1 -> 512 + 256 = 768
        self.up3_1 = up(init_channel * 16, init_channel * 24, attention_up=True, kernel_size=patch_size // 16)

        self.up0_2 = up1(init_channel * 6, init_channel * 10, attention_up=True, kernel_size=patch_size // 2)    # 0.2 -> 384 + 64 + 192 = 640
        self.up1_2 = up1(init_channel * 12, init_channel * 20, attention_up=True, kernel_size=patch_size // 4)   # 1.2 -> 768 + 128 + 384  = 1280
        self.up2_2 = up1(init_channel * 24, init_channel * 40, attention_up=True, kernel_size=patch_size // 8)

        self.up0_3 = up2(init_channel * 20, init_channel * 34, attention_up=True,kernel_size=patch_size // 2)            # 0.3 ->  1280 + 64 + 192 + 640 = 2176( out of memory)
        self.up1_3 = up2(init_channel * 34, init_channel * 68, attention_up=True, kernel_size=patch_size // 4)

        self.up0_4 = up3(init_channel * 68, init_channel * 116, attention_up=False, kernel_size=patch_size // 2)


        self.out0_1 = outconv(init_channel * 3, n_classes)
        self.out0_2 = outconv(init_channel * 10, n_classes)
        self.out0_3 = outconv(init_channel * 34, n_classes)
        self.out0_4 = outconv(init_channel * 116, n_classes)


    def forward(self, x):

        x0_0 = self.inc(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        x0_1 = self.up0_1(x1_0, x0_0)
        x1_1 = self.up1_1(x2_0, x1_0)
        x2_1 = self.up2_1(x3_0, x2_0)
        x3_1 = self.up3_1(x4_0, x3_0)

        x0_2 = self.up0_2(x0_0, x0_1, x1_1)
        x1_2 = self.up1_2(x1_0, x1_1, x2_1)
        x2_2 = self.up2_2(x2_0, x2_1, x3_1)

        x0_3 = self.up0_3(x0_0, x0_1, x0_2, x1_2)
        x1_3 = self.up1_3(x1_0, x1_1, x1_2, x2_2)

        x0_4 = self.up0_4(x0_0, x0_1, x0_2, x0_3, x1_3)

        x0_1 = self.out0_1(x0_1)
        x0_2 = self.out0_2(x0_2)
        x0_3 = self.out0_3(x0_3)
        x0_4 = self.out0_4(x0_4)


        #return F.softmax(x0_1, dim=1), F.softmax(x0_2, dim=1), F.softmax(x0_3, dim=1), F.softmax(x0_4, dim=1)
        return F.sigmoid(x0_1), F.sigmoid(x0_2), F.sigmoid(x0_3), F.sigmoid(x0_4)


class Unet_GCN(nn.Module):
    def __init__(self, n_channels, n_classes, init_channel=32):
        super(Unet_GCN, self).__init__()
        self.init_channel = init_channel
        self.inc = inconv(n_channels, init_channel)
        self.down1 = down(init_channel, init_channel * 2)
        self.down2 = down(init_channel * 2, init_channel * 4)
        self.down3 = down(init_channel * 4, init_channel * 8)

        #Unet++,
        self.up0_1 = up(init_channel * 2, init_channel * 3, True) # 0.1 -> 128 + 64 = 192
        self.up1_1 = up(init_channel * 4, init_channel * 6, True) # 1.1 -> 256 + 128 = 384
        self.up2_1 = up(init_channel * 8, init_channel * 12, True) # 2.1 -> 512 + 256 = 768
        self.up0_2 = up1(init_channel * 6, init_channel * 10)    # 0.2 -> 384 + 64 + 192 = 640
        self.up1_2 = up1(init_channel * 12, init_channel * 20)   # 1.2 -> 768 + 128 + 384  = 1280
        self.up0_3 = up2(init_channel * 20, init_channel * 34)            # 0.3 ->  1280 + 64 + 192 + 640 = 2176( out of memory)

        self.outc0_1 = outconv(init_channel * 3, n_classes)
        self.outc0_2 = outconv(init_channel * 10, n_classes)
        self.outc0_3 = outconv(init_channel * 34, n_classes, CAM_outout=False)


    def forward(self, x):

        x0_0 = self.inc(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)

        x0_1 = self.up0_1(x1_0, x0_0)
        x1_1 = self.up1_1(x2_0, x1_0)
        x2_1 = self.up1_1(x3_0, x2_0)

        x0_2 = self.up0_2(x0_0, x0_1, x1_1)
        x1_2 = self.up1_2(x1_0, x1_1, x2_1)
        x0_3 = self.up0_3(x0_0, x0_1, x0_2, x1_2)

        x0_1 = self.outc0_1(x0_1)
        x0_2 = self.outc0_2(x0_2)
        x0_3 = self.outc0_3(x0_3)


        return x0_1, x0_2, x0_3


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256, single_up=False, bilinear=False, GCN_module=False)
#         self.up2 = up(512, 128, single_up=False, bilinear=False, GCN_module=False)
#         self.up3 = up(256, 64, single_up=False, bilinear=False, GCN_module=False)
#         self.up4 = up(128, 64, single_up=False, bilinear=False, GCN_module=False)
#         self.outc = outconv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         return F.softmax(x, dim=1)
#         #return F.sigmoid(x)
