
import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, mask=None):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
              pred (batch x c x h x w)
              gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        if mask is not None:
            pos_inds = pos_inds * mask
            neg_inds = neg_inds * mask

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.temp = temperature

    def forward(self, features, t=None):
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)  # L2 normalization
        # 先单位向量化,再计算内积,此时的结果即为cosine similarity
        if t != None:
            logits = features_norm.mm(
                features_norm.t()) / t  # temp---> softmax temperature  # n * n, each value means similarity.
        else:
            logits = features_norm.mm(
                features_norm.t()) / self.temp  # temp---> softmax temperature  # n * n, each value means similarity.
        targets = torch.arange(n, dtype=torch.long).to(features.device)

        # 首先logits经过softmax处理, 再求-y * log(y)的sum和（带log的为预测值）,交叉熵由于带log, 且概率不会超过1, 通常结果为负数, 直接用来当作损失函数的话0为
        loss = F.cross_entropy(logits, targets, reduction='sum')
        return loss








