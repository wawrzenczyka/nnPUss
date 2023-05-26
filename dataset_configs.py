from typing import TypeVar

from torchvision import transforms

from dataset import MNIST_PU_CC, MNIST_PU_SS, TwentyNews_PU_CC, TwentyNews_PU_SS


class DatasetConfig:
    def __init__(self, name, dataset_class, positive_labels, normalization):
        self.name = name
        self.DatasetClass = dataset_class
        self.positive_labels = positive_labels
        self.normalization = normalization


class DatasetConfigs:
    MNIST_CC = DatasetConfig(
        "MNIST CC",
        MNIST_PU_CC,
        positive_labels=[0, 2, 4, 6, 8],
        normalization=transforms.Normalize((0.1307,), (0.3081,)),
    )
    MNIST_SS = DatasetConfig(
        "MNIST SS",
        MNIST_PU_SS,
        positive_labels=[0, 2, 4, 6, 8],
        normalization=transforms.Normalize((0.1307,), (0.3081,)),
    )

    TwentyNews_CC = DatasetConfig(
        "20News CC",
        TwentyNews_PU_CC,
        positive_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        normalization=transforms.Normalize((-0.0004,), (0.0510,)),
    )
    TwentyNews_SS = DatasetConfig(
        "20News SS",
        TwentyNews_PU_SS,
        positive_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        normalization=transforms.Normalize((-0.0004,), (0.0510,)),
    )
