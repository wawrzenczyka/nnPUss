from abc import abstractmethod

import torch
from torch import nn


class _PULoss(nn.Module):
    def __init__(
        self,
        prior,
        loss=(lambda x: torch.sigmoid(-x)),
        gamma=1,
        beta=0,
        nnPU=False,
        single_sample=False,
    ):
        super().__init__()
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

    def forward(self, x, target, test=False):
        assert x.shape == target.shape
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if x.is_cuda:
            self.min_count = self.min_count.cuda()
            self.prior = self.prior.cuda()
        n_positive, n_unlabeled = torch.max(
            self.min_count, torch.sum(positive)
        ), torch.max(self.min_count, torch.sum(unlabeled))
        n = n_positive + n_unlabeled

        y_positive = self.loss_func(x)
        y_unlabeled = self.loss_func(-x)

        positive_risk = self.prior * torch.sum(positive * y_positive) / n_positive
        negative_risk_c1_cc = torch.sum(unlabeled * y_unlabeled) / n_unlabeled
        negative_risk_c1_ss = torch.sum(y_unlabeled) / n
        negative_risk_c2 = self.prior * torch.sum(positive * y_unlabeled) / n_positive

        if not self.single_sample:
            negative_risk = negative_risk_c1_cc - negative_risk_c2
        else:
            negative_risk = negative_risk_c1_ss - negative_risk_c2

        diagnostic_vals = {
            "Labeled component": 0,
            "Whole distribution component CC": 0,
            "Whole distribution component SS": 0,
            "PU SCAR correction": 0,
        }

        if self.nnPU and negative_risk < -self.beta:
            return -self.gamma * negative_risk, diagnostic_vals
        else:
            return positive_risk + negative_risk, diagnostic_vals

    @property
    @abstractmethod
    def name():
        raise NotImplementedError("Implement in subclasses")


class nnPUccLoss(_PULoss):
    name = "nnPUcc"

    def __init__(
        self,
        prior,
        loss=lambda x: torch.sigmoid(-x),
        gamma=1,
        beta=0,
    ):
        super().__init__(prior, loss, gamma, beta, nnPU=True, single_sample=False)


class nnPUssLoss(_PULoss):
    name = "nnPUss"

    def __init__(
        self,
        prior,
        loss=lambda x: torch.sigmoid(-x),
        gamma=1,
        beta=0,
    ):
        super().__init__(prior, loss, gamma, beta, nnPU=True, single_sample=True)


class uPUccLoss(_PULoss):
    name = "uPUcc"

    def __init__(
        self,
        prior,
        loss=lambda x: torch.sigmoid(-x),
        gamma=1,
        beta=0,
    ):
        super().__init__(prior, loss, gamma, beta, nnPU=False, single_sample=False)


class uPUssLoss(_PULoss):
    name = "uPUss"

    def __init__(
        self,
        prior,
        loss=lambda x: torch.sigmoid(-x),
        gamma=1,
        beta=0,
    ):
        super().__init__(prior, loss, gamma, beta, nnPU=False, single_sample=True)
