# %%
import torch
from torch.utils import data
from torchvision.datasets import MNIST


class SCARLabeler:
    def __init__(
        self, positive_labels=[1, 3, 5, 7, 9], label_frequency=0.5, NEGATIVE_LABEL=-1
    ) -> None:
        self.positive_labels = positive_labels
        self.label_frequency = label_frequency
        self.NEGATIVE_LABEL = NEGATIVE_LABEL

    def relabel(self, labels, is_train):
        y = torch.where(
            torch.isin(labels, torch.tensor(self.positive_labels)),
            1,
            self.NEGATIVE_LABEL,
        )

        if is_train:
            labeling_condition = torch.rand_like(y, dtype=float) < self.label_frequency
            s = self.scar_targets = torch.where(
                y == 1,
                torch.where(
                    labeling_condition,
                    1,
                    self.NEGATIVE_LABEL,
                ),
                self.NEGATIVE_LABEL,
            )
        else:
            s = self.scar_targets = torch.empty_like(y)

        return y, s


class PU_MNIST_SS(MNIST):
    def __init__(
        self,
        root,
        scar_labeler: SCARLabeler,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.binary_targets, self.scar_targets = scar_labeler.relabel(
            self.targets, train
        )

    def __getitem__(self, idx):
        input, _ = super().__getitem__(idx)
        target = self.binary_targets[idx]
        label = self.scar_targets[idx]
        return input, target, label

    def __len__(self) -> int:
        return super().__len__()

    def get_prior(self):
        if self.train:
            return torch.mean((self.binary_targets == 1).float())
        else:
            return None


class _SSToCCDatasetConverter:
    def __init__(self) -> None:
        pass

    def convert(self, dataset: data.Dataset, is_train: bool):
        if is_train:
            positive_idx = torch.where(dataset.scar_targets == 1)[0]
            dataset.data = torch.cat([dataset.data, dataset.data[positive_idx]])
            dataset.targets = torch.cat(
                [dataset.targets, dataset.targets[positive_idx]]
            )
            dataset.binary_targets = torch.cat(
                [dataset.binary_targets, dataset.binary_targets[positive_idx]]
            )
            dataset.scar_targets = torch.cat(
                [
                    torch.zeros_like(dataset.scar_targets),
                    dataset.scar_targets[positive_idx],
                ]
            )


class PU_MNIST_CC(PU_MNIST_SS):
    def __init__(
        self,
        root,
        scar_labeler: SCARLabeler,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(
            root,
            scar_labeler=scar_labeler,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.prior = super().get_prior()

        _SSToCCDatasetConverter().convert(self, train)

    def get_prior(self):
        return self.prior


class PN_MNIST(MNIST):
    def __getitem__(self, i):
        input, target = super(PN_MNIST, self).__getitem__(i)
        if target % 2 == 0:
            target = torch.tensor(1)
        else:
            target = torch.tensor(-1)

        return input, target


# %%
