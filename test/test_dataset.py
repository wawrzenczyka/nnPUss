import sys

import numpy as np
import torch

sys.path.append(".")
from src.nnPUss.dataset import MNIST_PU_CC, MNIST_PU_SS, SCAR_SS_Labeler


def test_mnist_ss_prior():
    c = 0.3
    EPS = 0.05
    dataset = MNIST_PU_SS("data", SCAR_SS_Labeler(label_frequency=c), download=True)
    assert np.abs(dataset.get_prior() - 0.5) < EPS


def test_mnist_ss_proportions():
    c = 0.3
    EPS = 0.05
    dataset = MNIST_PU_SS("data", SCAR_SS_Labeler(label_frequency=c), download=True)

    empirical_c = torch.sum(dataset.pu_targets == 1) / torch.sum(
        dataset.binary_targets == 1
    )
    assert np.abs(empirical_c - c) < EPS

    empirical_pi_l = torch.sum(dataset.pu_targets == 1) / len(dataset.pu_targets)
    assert np.abs(empirical_pi_l - c * dataset.get_prior()) < EPS


def test_mnist_cc_prior():
    c = 0.3
    EPS = 0.05
    dataset = MNIST_PU_CC("data", SCAR_SS_Labeler(label_frequency=c), download=True)
    assert np.abs(dataset.get_prior() - 0.5) < EPS


def test_mnist_cc_proportions():
    c = 0.3
    EPS = 0.05
    dataset = MNIST_PU_CC("data", SCAR_SS_Labeler(label_frequency=c), download=True)

    n = 60_000
    assert np.abs(len(dataset) - n) < EPS * n

    empirical_c = torch.sum(dataset.pu_targets == 1) / torch.sum(
        dataset.binary_targets == 1
    )
    assert np.abs(empirical_c - c) < EPS
