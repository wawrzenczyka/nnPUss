# %%
import os

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset, random_split
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
            s = self.scar_targets = torch.zeros_like(y)

        return y, s


class SingleSampleDataset(Dataset):
    def __init__(self, ss_labeler: SCAR_SS_Labeler, train: bool) -> None:
        self.ss_labeler = ss_labeler
        self.train = train

    def _convert_labels_to_pu(self):
        self.binary_targets, self.pu_targets = self.ss_labeler.relabel(
            self.targets, self.train
        )

    def __len__(self):
        return len(self.pu_targets)

    def __getitem__(self, idx):
        input = self.data[idx]
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

        n = len(self.targets)
        c = self.ss_labeler.label_frequency
        A = 1 / (1 - c + c * self.prior)

        P_sampling_probability = A * c
        U_sampling_probability = A * (1 - c)

        if self.train:
            positive_idx = torch.where(self.binary_targets == 1)[0]
            p_sampling_condition = (
                torch.rand_like(positive_idx, dtype=float) < P_sampling_probability
            )
            positive_labeled_idx = positive_idx[p_sampling_condition]

            u_sampling_condition = (
                torch.rand_like(self.targets, dtype=float) < U_sampling_probability
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

    def __len__(self):
        return len(self.pu_targets)

    def __getitem__(self, idx):
        input = self.data[idx]
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
        # transform=None,
        # target_transform=None,
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
            # transform=transform,
            # target_transform=target_transform,
            download=download,
        )
        self.data = self.data / 255.0
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
        self.data = self.data / 255.0
        self._convert_labels_to_pu()


class MNIST_PU_SS_Joined(SingleSampleDataset, MNIST):
    def __init__(
        self,
        root,
        scar_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        random_seed=42,
    ):
        SingleSampleDataset.__init__(
            self,
            ss_labeler=scar_labeler,
            train=train,
        )

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_mnist = MNIST(
            root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        test_mnist = MNIST(
            root,
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.data = torch.cat([train_mnist.data, test_mnist.data])
        self.targets = torch.cat([train_mnist.targets, test_mnist.targets])

        generator = torch.Generator().manual_seed(random_seed)
        train_idx, test_idx = random_split(
            range(len(self.targets)), [60_000, 10_000], generator=generator
        )
        if train:
            self.data = self.data[train_idx]
            self.targets = self.targets[train_idx]
        else:
            self.data = self.data[test_idx]
            self.targets = self.targets[test_idx]

        self._convert_labels_to_pu()


class MNIST_PU_CC_Joined(CaseControlDataset, MNIST):
    def __init__(
        self,
        root,
        scar_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        random_seed=42,
    ):
        CaseControlDataset.__init__(
            self,
            ss_labeler=scar_labeler,
            train=train,
        )

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_mnist = MNIST(
            root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        test_mnist = MNIST(
            root,
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.data = torch.cat([train_mnist.data, test_mnist.data])
        self.targets = torch.cat([train_mnist.targets, test_mnist.targets])

        generator = torch.Generator().manual_seed(random_seed)
        train_idx, test_idx = random_split(
            range(len(self.targets)), [60_000, 10_000], generator=generator
        )
        if train:
            self.data = self.data[train_idx]
            self.targets = self.targets[train_idx]
        else:
            self.data = self.data[test_idx]
            self.targets = self.targets[test_idx]

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


class SyntheticDataset(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        size=1000,
        random_seed=42,
    ):
        self.transform = transform
        self.target_transform = target_transform

        if not train:
            random_seed = random_seed + 42
        generator = torch.Generator().manual_seed(random_seed)

        n_pos = int(0.8 * size)
        X_pos = torch.randn((n_pos, 2), generator=generator) + torch.tensor([[2, 0]])
        X_neg = torch.randn((size - n_pos, 2), generator=generator) + torch.tensor(
            [[-2, 0]]
        )

        X = torch.cat([X_pos, X_neg])
        y = torch.cat([torch.ones(len(X_pos)), torch.zeros(len(X_neg))])

        self.data, self.targets = X, y

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


class Synthetic_PU_SS(SingleSampleDataset, SyntheticDataset):
    def __init__(
        self,
        root,
        ss_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        size=1000,
        random_seed=42,
    ):
        SingleSampleDataset.__init__(
            self,
            ss_labeler=ss_labeler,
            train=train,
        )
        SyntheticDataset.__init__(
            self,
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            size=size,
            random_seed=random_seed,
        )
        self._convert_labels_to_pu()


class Synthetic_PU_CC(CaseControlDataset, SyntheticDataset):
    def __init__(
        self,
        root,
        ss_labeler: SCAR_SS_Labeler,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        size=1000,
        random_seed=42,
    ):
        CaseControlDataset.__init__(
            self,
            ss_labeler=ss_labeler,
            train=train,
        )
        SyntheticDataset.__init__(
            self,
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            size=size,
            random_seed=random_seed,
        )
        self._convert_labels_to_pu()


# %%
