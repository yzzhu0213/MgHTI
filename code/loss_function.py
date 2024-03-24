import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.logits = logits
        self.reduce = reduce
        self.gamma = 2
        self.alpha = 0.5

    def forward(self, pred, target):
        pred = pred.reshape(-1, 1)
        target = target.reshape(-1, 1)

        pred = pred.clamp(min=0.00001, max=0.99999)

        pred_cat = torch.cat((pred.log(), (1 - pred).log()), dim=1)

        pt = torch.cat((1 - pred, pred), dim=1)

        target_cat = torch.cat((self.alpha * target, (1 - self.alpha) * (1 - target)), dim=1)

        batch_loss = -1 * (target_cat * (torch.pow(pt, self.gamma)) * pred_cat)

        batch_loss_sum = batch_loss.sum(axis=0)
        target_n = target_cat.sum(0)

        if target_n[0] == 0:
            loss = batch_loss_sum[1] / target_n[1]
        else:
            batch_loss_sum = batch_loss_sum / target_n
            loss = batch_loss_sum.sum()
        return loss
