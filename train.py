import sys
import os
from optparse import OptionParser
from PIL import Image
import numpy as np
import random
import torchvision.models as models

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch import optim
from unet import UNet, Nested_unet, Unet_GCN
from utils.extract_patches import get_data_training
from FocalLoss import *
from dice_loss import *
from utils.extract_patches import my_PreProc, paint_border, extract_ordered, recompone, kill_border, recompone_overlap, pred_only_FOV, get_data_testing, get_data_testing_overlap
from utils.preprocessing import mask_to_image, plot_img_and_mask, pred_to_imgs, load_hdf5, visualize, group_images


dir_checkpoint = 'E:\\h5_data\\'

N_subimgs = 19000
start_record_epoch = 150
patch_height = 48
patch_width = 48
stride_height = 5
stride_width = 5
Imgs_to_test = 20
DRIVE_test_imgs_original = dir_checkpoint + 'DRIVE_dataset_imgs_test.hdf5'
DRIVE_test_groudTruth = dir_checkpoint + 'DRIVE_dataset_groundTruth_test.hdf5'
DRIVE_test_borderMask = dir_checkpoint + 'DRIVE_dataset_borderMasks_test.hdf5'



class TrainDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs, patches_masks_train):
        self.imgs = patches_imgs
        self.masks = patches_masks_train

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        tmp = self.masks[idx]
        tmp = np.squeeze(tmp, 0)
        return torch.from_numpy(self.imgs[idx, ...]).float(), torch.from_numpy(tmp).long()

class TestDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs):
        self.imgs = patches_imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx, ...]).float()

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              gpu=False):

    ratio = 0.1

    #============ Load the data and divided in patches
    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original=dir_checkpoint + 'DRIVE_dataset_imgs_train.hdf5',
        DRIVE_train_groudTruth=dir_checkpoint +
        'DRIVE_dataset_groundTruth_train.hdf5',  # masks
        patch_height=patch_height,
        patch_width=patch_width,
        N_subimgs=N_subimgs,
        # select the patches only inside the FOV  (default == True)
        inside_FOV=False
    )

    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        DRIVE_test_imgs_original=DRIVE_test_imgs_original,
        DRIVE_test_groudTruth=DRIVE_test_groudTruth,
        Imgs_to_test=Imgs_to_test,
        patch_height=patch_height,
        patch_width=patch_width,
        stride_height=stride_height,
        stride_width=stride_width
    )

    val_ind = random.sample(range(patches_masks_train.shape[0]), int(
        np.floor(ratio * patches_masks_train.shape[0])))

    train_ind = set(range(patches_masks_train.shape[0])) - set(val_ind)
    train_ind = list(train_ind)

    train_set = TrainDataset(
        patches_imgs_train[train_ind, ...], patches_masks_train[train_ind, ...])
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    val_set = TrainDataset(
        patches_imgs_train[val_ind, ...], patches_masks_train[val_ind, ...])
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    test_imgs_bordermask = load_hdf5(DRIVE_test_borderMask)
    test_set = TestDataset(patches_imgs_test)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    del patches_imgs_train, patches_imgs_test

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, str(save_cp), str(gpu)))

    min_loss = float('inf')
    max_auc = 0
    best_epoch = 0
    writer = SummaryWriter()

    # optimizer = optim.Adam(net.parameters(),
    #                       lr=lr,
    #                       weight_decay=0.0005)
    #scheduler = lr_scheduler.MultiStepLR
    optimizer = optim.SGD(net.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=0.0005)
    criterion_1 = FocalLoss()
    criterion_2 = nn.BCELoss() #sigmoid
    criterion_3 = nn.CrossEntropyLoss() #softmax
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70])
    # feature_map = []

    # def hook_feature(module, input, output):
    #     feature_map.append(output)
    # net._modules.get('up0_3').register_forward_hook(hook_feature)



    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        #lr = pow(1 - epoch / epochs, 0.9)
        # train_batch_weight = torch.empty((batch_size, 1, 32 * 34)).cuda()  # (bn, 1, up3_channel)
        # val_batch_weight = torch.empty((batch_size, 1, 32 * 34)).cuda()  # (bn, 1, up3_channel)

        train_loss = 0
        val_loss = 0
        #scheduler.step(epoch)


        for batch_idx, (inputs, targets) in enumerate(train_loader): # train
            net.train(True)
            if gpu:
                inputs = inputs.cuda()
                targets = targets.long().cuda()

            # if(inputs.shape[0] != batch_size):
            #     train_batch_weight = torch.empty((inputs.shape[0], 1, 32 * 34)).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            out1, out2, out3, out4 = net(inputs)
            loss1 = criterion_2(out1, targets.float())
            loss2 = criterion_2(out2, targets.float())
            loss3 = criterion_2(out3, targets.float())
            loss4 = criterion_2(out4, targets.float())
            loss = loss1 + loss2 + loss3 + loss4
            #loss2 = criterion2(out_cam.view(out_cam.shape[0], -1), targets.view(targets.shape[0], -1).float())
            #loss2 = criterion2(out_cam, targets.float())
            train_loss = train_loss + loss1.item() + loss2.item() + loss3.item() + loss4.item()
            loss.backward()
            optimizer.step()
            #feature_map.clear()

        print('Epoch finished ! Training Loss: {}'.format(train_loss / len(train_loader)))
        writer.add_scalars('data/loss', {'loss': train_loss / len(train_loader)}, epoch)

        net.eval() # val mode
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                if gpu:
                    inputs = inputs.cuda()
                    targets = targets.long().cuda()
#########################################################################################################################(val Dataset)
                out1, out2, out3, out4 = net(inputs)
                loss1 = criterion_2(out1, targets.float())
                loss2 = criterion_2(out2, targets.float())
                loss3 = criterion_2(out3, targets.float())
                loss4 = criterion_2(out4, targets.float())
                loss = loss1 + loss2 + loss3 + loss4
                #loss2 = criterion2(out_cam.view(out_cam.shape[0], -1), targets.view(targets.shape[0], -1).float())
                #loss2 = criterion2(out_cam, targets.float())
                val_loss = val_loss + loss1.item() + loss2.item() + loss3.item() + loss4.item()
                #val_loss += loss.item()
                #if save_cp & (val_loss < min_loss) & epoch > 50:
                if save_cp and epoch > start_record_epoch:
                    #min_loss = val_loss
                    torch.save(net.state_dict(), dir_checkpoint + 'model\\' + str(epoch) + '.pth')
        print('Epoch finished ! Val Loss: {}'.format(val_loss / len(val_loader)))
        writer.add_scalars('data/loss', {'loss': val_loss / len(val_loader)}, epoch)

        if(epoch > start_record_epoch) and (val_loss / len(val_loader) < min_loss):
            min_loss = val_loss / len(val_loader)
            preds = []  # patch
            net.eval() # eval mode
            with torch.no_grad():
                for batch_idx, inputs in enumerate(test_loader):
                    if gpu:
                        inputs = inputs.cuda()
    #########################################################################################################################(test Dataset)
                    out1, out2, out3, out4 = net(inputs)
                    # outputs = F.softmax(out, dim=1)
                    #softmax need
                    # outputs = out4.permute(0, 2, 3, 1) #[BN, H, W, C]
                    # outputs = outputs.view(-1, outputs.shape[1] * outputs.shape[2], 2)

                    outputs = out4.data.cpu().numpy()
                    preds.append(outputs)
                    del outputs
                predictions = np.concatenate(preds, axis=0)
                #predictions = torch.cat(preds, 0)
                preds.clear()
                #predictions = pred_to_imgs(predictions, patch_height, patch_width, "original") # only work in softmax(sigmoid dont need)
                print("Predictions finished")
                pred_imgs = recompone_overlap(predictions, new_height, new_width, stride_height, stride_width)
                del predictions
                gtruth_masks = masks_test  # ground truth masks
                pred_imgs = pred_imgs[:, :, 0:masks_test.shape[2], 0:masks_test.shape[3]]
                pred_imgs = kill_border(pred_imgs, test_imgs_bordermask)
                y_scores1, y_true1 = pred_only_FOV(pred_imgs, gtruth_masks, test_imgs_bordermask)
                fpr1, tpr1, thresholds = roc_curve((y_true1), y_scores1)
                AUC_ROC1 = roc_auc_score(y_true1, y_scores1)
                print('Auc: {}'.format(AUC_ROC1))
                if AUC_ROC1 > max_auc:
                    max_auc = AUC_ROC1
                    best_epoch = epoch
                # if save_cp & (AUC_ROC1 > auc_value):
                #     auc_value = AUC_ROC1
                #     torch.save(net.state_dict(),
                #             dir_checkpoint + save_name)

#########################################################################################################################
                # if(inputs.shape[0] != batch_size):
                #     val_batch_weight = np.empty((inputs.shape[0], 1, 32 * 34))

                # out1, out2, out3 = net(inputs)
                # loss1 = criterion(out1, targets)
                # loss2 = criterion(out2, targets)
                # loss3 = criterion_bce(out3, targets.float())

                # params = list(net.parameters())
                # weight = torch.squeeze(params[-2])  # (2, channel)
                #val_batch_weight[:, :, :] = weight[1, :].clone() #假定这里1是vessel权重

                #cam
                # # multi-dimension  multi
                # bz, nc, h, w = feature_map[0].shape
                # feature_batch = feature_map[0].reshape((bz, nc, h * w))
                # cam = torch.einsum('ijk,ikl->ijl', [val_batch_weight, feature_batch])
                # cam = torch.reshape(cam, (out3.shape[0], 1, h, w))
                # cam = cam - torch.min(cam)
                # cam_img = cam / torch.max(cam)
                # loss4 = criterion_bce(cam_img.cuda(), torch.unsqueeze(targets, 1).float())

                # loss = loss1 + loss2 + loss3
                # val_loss += loss.item()
                # feature_map.clear()

        #print('Epoch finished ! Val Loss: {}'.format(val_loss / len(val_loader)))
        #writer.add_scalars('data/loss', {'loss': val_loss / len(val_loader)}, epoch)


        # if 1:
        #     val_dice = eval_net(net, val, gpu)
        #     writer.add_scalars('data/loss', {'dice': val_dice}, epoch)
        #     print('Validation Dice Coeff: {}'.format(val_dice))

        # if save_cp & (val_loss / len(val_loader) < min_loss):
        #     min_loss = val_loss / len(val_loader)
        #     torch.save(net.state_dict(),
        #             dir_checkpoint + save_name)
    writer.close()
    print('Best epoch: {}'.format(best_epoch))
    print('Best Auc: {}'.format(max_auc))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = Nested_unet(n_channels=1, n_classes=1)
    model = models.resnet50(pretrained=True)

    pretrained_dict = model.state_dict().copy()
    model_dict = net.state_dict().copy()

    #load part of pretrained model
    #model_dict['inc.conv.conv.4.weight'] = pretrained_dict['layer1.0.conv2.weight']
    #net.load_state_dict(model_dict)

    # # inilization
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.xavier_normal_(m.weight.data)
    #     # elif isinstance(m, nn.BatchNorm2d):
    #     #     nn.init.constant_(m.weight.data, 1)
    #     #     nn.init.constant_(m.bias.data, 0)

    if args.load:
        net.load_state_dict(torch.load(dir_checkpoint + '\\model\\184.pth'))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
