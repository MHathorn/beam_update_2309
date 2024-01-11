
from fastai.vision.all import *


# Custom loss functions
class CombinedLoss:
    """
    Dice and Focal combined
    """

    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):    return x.argmax(dim=self.axis)

    def activation(self, x): return F.softmax(x, dim=self.axis)

class CombinedCrossDiceLoss:
    """
    Cross Entropy and Dice combined
    """

    def __init__(self, pixel_weights=None, smooth=1., alpha=1.):
        store_attr()
        self.cross_entropy_loss = CrossEntropyLossFlat(weight=pixel_weights)
        self.dice_loss = DiceLoss(smooth)

    def __call__(self, pred, targ):
        return self.cross_entropy_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)



class DualFocalLoss(nn.Module):
    """
    This loss is proposed in this paper: https://arxiv.org/abs/1909.11932
    """

    def __init__(self, ignore_lb=255, eps=1e-5, reduction='mean'):
        super(DualFocalLoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.eps = eps
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, logits, label):
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1).detach()

        pred = torch.softmax(logits, dim=1)
        loss = -torch.log(self.eps + 1. - self.mse(pred, lb_one_hot)).sum(dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss