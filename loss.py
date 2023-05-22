import torch
from torch import nn


class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(
        self,
        prior,
        loss=(lambda x: torch.sigmoid(-x)),
        gamma=1,
        beta=0,
        nnPU=False,
        single_sample=False,
    ):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss  # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.single_sample = single_sample
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.0)

    def forward(self, inp, target, test=False):
        assert inp.shape == target.shape
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if inp.is_cuda:
            self.min_count = self.min_count.cuda()
            self.prior = self.prior.cuda()
        n_positive, n_unlabeled = torch.max(
            self.min_count, torch.sum(positive)
        ), torch.max(self.min_count, torch.sum(unlabeled))
        n = n_positive + n_unlabeled

        y_positive = self.loss_func(positive * inp) * positive
        y_positive_inv = self.loss_func(-positive * inp) * positive
        y_unlabeled = self.loss_func(-unlabeled * inp) * unlabeled

        if not self.single_sample:
            positive_risk = self.prior * torch.sum(y_positive) / n_positive
            negative_risk = (
                -self.prior * torch.sum(y_positive_inv) / n_positive
                + torch.sum(y_unlabeled) / n_unlabeled
            )
        else:
            positive_risk = self.prior * torch.sum(positive * y_positive) / n_positive
            negative_risk = (
                (n_unlabeled / n)
                * (1 / n_unlabeled)
                * torch.sum(unlabeled * y_unlabeled)
            ) - (
                torch.maximum(torch.tensor(0), self.prior - n_positive / n)
                * (1 / n_positive)
                * torch.sum(positive * y_unlabeled)
            )

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk
