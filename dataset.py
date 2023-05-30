# %%
import os

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST


class SCAR_SS_Labeler:
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
    def __init__(self, ss_labeler: SCAR_SS_Labeler, train: bool) -> None:
        self.ss_labeler = ss_labeler
        self.train = train

    def _convert_labels_to_pu(self):
        self.binary_targets, self.pu_targets = self.ss_labeler.relabel(
            self.targets, self.train
        )

    def __getitem__(self, idx):
        input, _ = super().__getitem__(idx)
        target = self.binary_targets[idx]
        label = self.pu_targets[idx]
        return input, target, label

    def get_prior(self):
        if self.train:
            return torch.mean((self.binary_targets == 1).float())
        else:
            return None


class CaseControlDataset(Dataset):
    def __init__(
        self, ss_labeler: SCAR_SS_Labeler, train: bool, NEGATIVE_LABEL: int = -1
    ) -> None:
        self.ss_labeler = ss_labeler
        self.train = train
        self.NEGATIVE_LABEL = NEGATIVE_LABEL

    def _convert_labels_to_pu(self):
        self.binary_targets, self.pu_targets = self.ss_labeler.relabel(
            self.targets, self.train
        )
        self.prior = torch.mean((self.binary_targets == 1).float())

        if self.train:
            positive_labeled_idx = torch.where(self.pu_targets == 1)[0]

            u_sampling_condition = (
                torch.rand_like(self.targets, dtype=float)
                < 1 - self.ss_labeler.label_frequency
            )
            is_u_sample = torch.where(
                u_sampling_condition,
                True,
                False,
            )

            self.data = torch.cat(
                [
                    self.data[positive_labeled_idx],
                    self.data[is_u_sample],
                ]
            )
            self.targets = torch.cat(
                [
                    self.targets[positive_labeled_idx],
                    self.targets[is_u_sample],
                ]
            )
            self.binary_targets = torch.cat(
                [
                    self.binary_targets[positive_labeled_idx],
                    self.binary_targets[is_u_sample],
                ]
            )
            self.pu_targets = torch.cat(
                [
                    torch.ones_like(positive_labeled_idx),
                    self.NEGATIVE_LABEL * torch.ones(is_u_sample.int().sum().item()),
                ]
            )

    def __getitem__(self, idx):
        input, _ = super().__getitem__(idx)
        target = self.binary_targets[idx]
        label = self.pu_targets[idx]
        return input, target, label

    def get_prior(self):
        if self.train:
            return self.prior
        else:
            return None


class MNIST_PU_SS(SingleSampleDataset, MNIST):
    def __init__(
        self,
        root,
        scar_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        SingleSampleDataset.__init__(
            self,
            ss_labeler=scar_labeler,
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
        scar_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        CaseControlDataset.__init__(
            self,
            ss_labeler=scar_labeler,
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


class SentenceTransformersDataset(Dataset):
    def __init__(
        self,
        root,
        dataset_hub_path,
        dataset_name,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
    ):
        news_dataset = load_dataset(
            dataset_hub_path, cache_dir=os.path.join(root, dataset_name)
        )
        # embedding_model = SentenceTransformer("all-mpnet-base-v2")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.train = train
        if self.train:
            news_dataset = news_dataset["train"]
        else:
            news_dataset = news_dataset["test"]

        texts = news_dataset["text"]
        self.data = torch.from_numpy(embedding_model.encode(texts))
        self.targets = torch.tensor(news_dataset["label"])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # if self.train:
        return len(self.targets)

    def __getitem__(self, idx):
        # return super().__getitem__(idx)
        data, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            data = self.transform(data.numpy().reshape(1, *data.shape)).reshape(
                *data.shape
            )
        if self.target_transform is not None:
            target = self.transform(target)

        return data.numpy(), target


class TwentyNews(SentenceTransformersDataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
    ):
        super().__init__(
            root=root,
            dataset_hub_path="SetFit/20_newsgroups",
            dataset_name="20news",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class TwentyNews_PU_SS(SingleSampleDataset, TwentyNews):
    def __init__(
        self,
        root,
        ss_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
    ):
        SingleSampleDataset.__init__(
            self,
            ss_labeler=ss_labeler,
            train=train,
        )
        TwentyNews.__init__(
            self,
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self._convert_labels_to_pu()


class TwentyNews_PU_CC(CaseControlDataset, TwentyNews):
    def __init__(
        self,
        root,
        ss_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
    ):
        CaseControlDataset.__init__(
            self,
            ss_labeler=ss_labeler,
            train=train,
        )
        TwentyNews.__init__(
            self,
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self._convert_labels_to_pu()


class IMDB(SentenceTransformersDataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
    ):
        super().__init__(
            root=root,
            dataset_hub_path="imdb",
            dataset_name="IMDB",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class IMDB_PU_SS(SingleSampleDataset, IMDB):
    def __init__(
        self,
        root,
        ss_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
    ):
        SingleSampleDataset.__init__(
            self,
            ss_labeler=ss_labeler,
            train=train,
        )
        IMDB.__init__(
            self,
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self._convert_labels_to_pu()


class IMDB_PU_CC(CaseControlDataset, IMDB):
    def __init__(
        self,
        root,
        ss_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
    ):
        CaseControlDataset.__init__(
            self,
            ss_labeler=ss_labeler,
            train=train,
        )
        IMDB.__init__(
            self,
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self._convert_labels_to_pu()


# %%
