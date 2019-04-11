import torch
import torch.nn as nn
import torch.nn.functional as F
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py


class FocalLoss(nn.Module):

    def __init__(self, gamma=0.2, weight=0.25, sigmoid=True, size_average=True):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.sigmoid  = sigmoid
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):

        # target = target.contiguous().view(target.size(0), target.size(1), -1)
        # target = target.transpose(1, 2)
        # target = target.contiguous().view(-1, target.size(2)).squeeze()

        # input = input.contiguous().view(input.size(0), input.size(1), -1)
        # input = input.transpose(1, 2)
        # input = input.contiguous().view(-1, input.size(2)).squeeze()

        # compute the negative likelyhood
        if self.sigmoid:
            logpt = -F.binary_cross_entropy(input, target.float())
        else:
            logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt)**self.gamma) * logpt
        # loss = loss * self.weight
        # return loss

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
