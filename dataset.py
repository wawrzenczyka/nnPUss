# %%
import torch
from torch.utils.data import DataLoader, Dataset
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


class SingleSampleDataset(Dataset):
    def __init__(self, scar_labeler: SCARLabeler, train: bool) -> None:
        self.scar_labeler = scar_labeler
        self.train = train

    def _convert_labels_to_pu(self):
        self.binary_targets, self.scar_targets = self.scar_labeler.relabel(
            self.targets, self.train
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


class CaseControlDataset(Dataset):
    def __init__(self, scar_labeler: SCARLabeler, train: bool) -> None:
        self.scar_labeler = scar_labeler
        self.train = train

    def _convert_labels_to_pu(self):
        self.prior = super().get_prior()
        self.binary_targets, self.scar_targets = self.scar_labeler.relabel(
            self.targets, self.train
        )

        if self.train:
            positive_idx = torch.where(self.scar_targets == 1)[0]
            self.data = torch.cat([self.data, self.data[positive_idx]])
            self.targets = torch.cat([self.targets, self.targets[positive_idx]])
            self.binary_targets = torch.cat(
                [self.binary_targets, self.binary_targets[positive_idx]]
            )
            self.scar_targets = torch.cat(
                [
                    torch.zeros_like(self.scar_targets),
                    self.scar_targets[positive_idx],
                ]
            )

    def get_prior(self):
        return self.prior


class MNIST_PU_SS(SingleSampleDataset, MNIST):
    def __init__(
        self,
        root,
        scar_labeler: SCARLabeler,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        SingleSampleDataset.__init__(
            self,
            scar_labeler=scar_labeler,
            train=train,
        )
        MNIST.__init__(
            self,
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self._convert_labels_to_pu()


class MNIST_PU_CC(CaseControlDataset, MNIST):
    def __init__(
        self,
        root,
        scar_labeler: SCARLabeler,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        CaseControlDataset.__init__(
            self,
            scar_labeler=scar_labeler,
            train=train,
        )
        MNIST.__init__(
            self,
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self._convert_labels_to_pu()


# %%
from typing import Any

from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class TwentyNews(Dataset):
    def __init__(
        self,
        root,
        scar_labeler: SCARLabeler,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        news_dataset = load_dataset("SetFit/20_newsgroups")
        # embedding_model = SentenceTransformer("all-mpnet-base-v2")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        if train:
            news_dataset = news_dataset["train"]
        else:
            news_dataset = news_dataset["test"]

        self.data = news_dataset["text"]
        self.labels = news_dataset["label"]

    def __len__(self):
        if self.train:
            return len(self.news_dataset)

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
