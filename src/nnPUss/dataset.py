# %%
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST


class BinaryTargetTransformer:
    def __init__(
        self,
        included_classes: list[int],
        positive_classes: list[int],
        POSITIVE_CLASS=1,
        NEGATIVE_CLASS=-1,
    ) -> None:
        self._included_classes = included_classes
        self._positive_classes = positive_classes
        self._POSITIVE_CLASS = POSITIVE_CLASS
        self._NEGATIVE_CLASS = NEGATIVE_CLASS

    def transform(self, X, y):
        X = X[torch.isin(y, torch.tensor(self._included_classes))]
        y = y[torch.isin(y, torch.tensor(self._included_classes))]

        y = torch.where(
            torch.isin(y, torch.tensor(self._positive_classes)),
            self._POSITIVE_CLASS,
            self._NEGATIVE_CLASS,
        )
        return X, y


class PULabeler:
    def __init__(
        self, label_frequency: float, POSITIVE_LABEL=1, NEGATIVE_LABEL=-1
    ) -> None:
        self._label_frequency = label_frequency
        self._POSITIVE_LABEL = POSITIVE_LABEL
        self._NEGATIVE_LABEL = NEGATIVE_LABEL

    @property
    def label_frequency(self) -> float:
        return self._label_frequency

    @property
    def prior(self) -> float:
        raise NotImplementedError()

    def relabel(self, X, y):
        raise NotImplementedError()


class SCAR_SS_Labeler(PULabeler):
    def __init__(
        self, label_frequency: float, POSITIVE_LABEL=1, NEGATIVE_LABEL=-1
    ) -> None:
        super().__init__(
            label_frequency=label_frequency,
            POSITIVE_LABEL=POSITIVE_LABEL,
            NEGATIVE_LABEL=NEGATIVE_LABEL,
        )

    def relabel(self, X, y):
        self._prior = torch.mean((y == 1).float())

        labeling_condition = torch.rand_like(y, dtype=float) < self._label_frequency
        s = self.scar_targets = torch.where(
            y == 1,
            torch.where(
                labeling_condition,
                1,
                self._NEGATIVE_LABEL,
            ),
            self._NEGATIVE_LABEL,
        )

        return X, y, s

    @property
    def prior(self) -> float:
        return self._prior


class SCAR_CC_Labeler(PULabeler):
    def __init__(
        self, label_frequency: float, POSITIVE_LABEL=1, NEGATIVE_LABEL=-1
    ) -> None:
        super().__init__(
            label_frequency=label_frequency,
            POSITIVE_LABEL=POSITIVE_LABEL,
            NEGATIVE_LABEL=NEGATIVE_LABEL,
        )

    def relabel(self, X, y):
        self._prior = torch.mean((y == 1).float())

        n = len(y)
        c = self._label_frequency
        A = 1 / (1 - c + c * self._prior)

        P_samples_num = int(A * c * (self._prior * n))
        U_samples_num = int(A * (1 - c) * n)

        positive_idx = torch.where(y == 1)[0]
        selected_positive_idx = torch.multinomial(
            torch.ones_like(positive_idx, dtype=torch.float32),
            P_samples_num,
            replacement=True,
        )
        positive_labeled_idx = positive_idx[selected_positive_idx]

        unlabeled_idx = torch.multinomial(
            torch.ones_like(y, dtype=torch.float32), U_samples_num, replacement=True
        )

        X = torch.cat(
            [
                X[positive_labeled_idx],
                X[unlabeled_idx],
            ]
        )
        y = torch.cat(
            [
                y[positive_labeled_idx],
                y[unlabeled_idx],
            ]
        )
        s = torch.cat(
            [
                torch.ones_like(positive_labeled_idx),
                self._NEGATIVE_LABEL * torch.ones_like(unlabeled_idx),
            ]
        )

        return X, y, s

    @property
    def prior(self) -> float:
        return self._prior


class PUDatasetBase:
    train: bool
    target_transformer: BinaryTargetTransformer
    pu_labeler: PULabeler

    data: list
    targets: list
    pu_targets: list
    binary_targets: list

    def __len__(self):
        return len(self.pu_targets)

    def __getitem__(self, idx):
        input = self.data[idx]
        target = self.binary_targets[idx]
        label = self.pu_targets[idx]
        return input, target, label

    def _convert_to_pu_data(self):
        assert self.target_transformer is not None
        assert self.pu_labeler is not None

        assert self.data is not None
        assert self.targets is not None

        self.data, self.binary_targets = self.target_transformer.transform(
            self.data, self.targets
        )
        self.data, self.binary_targets, self.pu_targets = self.pu_labeler.relabel(
            self.data, self.binary_targets
        )
        return self.data, self.binary_targets, self.pu_targets

    def get_prior(self):
        assert self.train is not None

        if self.train:
            return self.pu_labeler.prior
        else:
            return None


class MNIST_PU(PUDatasetBase, MNIST):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(10), positive_classes=[1, 3, 5, 7, 9]
        ),
        train=True,
        download=False,
        random_seed=None,
    ):
        MNIST.__init__(
            self,
            root,
            train=train,
            download=download,
        )
        self.data = self.data / 255.0

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class DatasetSplitterMixin:
    def get_split_idx(
        self, dataset, split_type: Literal["train", "test"], random_seed, test_ratio=0.2
    ):
        assert (
            random_seed is not None
        ), "random_seed is necessary for a valid train / test split, please provide it"

        n_train = int(test_ratio * len(dataset))
        n_test = len(dataset) - n_train

        generator = torch.Generator().manual_seed(random_seed)
        train_idx, test_idx = random_split(
            range(len(dataset)), [n_train, n_test], generator=generator
        )

        if split_type == "train":
            return train_idx.indices
        return test_idx.indices


# class MNIST_PU_SS_Joined(SingleSampleDataset, MNIST):
#     def __init__(
#         self,
#         root,
#         scar_labeler: SCAR_SS_Labeler,
#         train=True,
#         transform=None,
#         target_transform=None,
#         download=False,
#         random_seed=42,
#     ):
#         SingleSampleDataset.__init__(
#             self,
#             pu_labeler=scar_labeler,
#             train=train,
#         )

#         self.transform = transform
#         self.target_transform = target_transform
#         self.train = train

#         train_mnist = MNIST(
#             root,
#             train=True,
#             transform=transform,
#             target_transform=target_transform,
#             download=download,
#         )
#         test_mnist = MNIST(
#             root,
#             train=False,
#             transform=transform,
#             target_transform=target_transform,
#             download=download,
#         )
#         self.data = torch.cat([train_mnist.data, test_mnist.data])
#         self.targets = torch.cat([train_mnist.targets, test_mnist.targets])

#         generator = torch.Generator().manual_seed(random_seed)
#         train_idx, test_idx = random_split(
#             range(len(self.targets)), [60_000, 10_000], generator=generator
#         )
#         if train:
#             self.data = self.data[train_idx]
#             self.targets = self.targets[train_idx]
#         else:
#             self.data = self.data[test_idx]
#             self.targets = self.targets[test_idx]

#         self._convert_labels_to_pu()


# class MNIST_PU_CC_Joined(CaseControlDatasetMixin, MNIST):
#     def __init__(
#         self,
#         root,
#         scar_labeler: SCAR_SS_Labeler,
#         train=True,
#         transform=None,
#         target_transform=None,
#         download=False,
#         random_seed=42,
#     ):
#         CaseControlDatasetMixin.__init__(
#             self,
#             pu_labeler=scar_labeler,
#             train=train,
#         )

#         self.transform = transform
#         self.target_transform = target_transform
#         self.train = train

#         train_mnist = MNIST(
#             root,
#             train=True,
#             transform=transform,
#             target_transform=target_transform,
#             download=download,
#         )
#         test_mnist = MNIST(
#             root,
#             train=False,
#             transform=transform,
#             target_transform=target_transform,
#             download=download,
#         )
#         self.data = torch.cat([train_mnist.data, test_mnist.data])
#         self.targets = torch.cat([train_mnist.targets, test_mnist.targets])

#         generator = torch.Generator().manual_seed(random_seed)
#         train_idx, test_idx = random_split(
#             range(len(self.targets)), [60_000, 10_000], generator=generator
#         )
#         if train:
#             self.data = self.data[train_idx]
#             self.targets = self.targets[train_idx]
#         else:
#             self.data = self.data[test_idx]
#             self.targets = self.targets[test_idx]

#         self._convert_labels_to_pu()


class SentenceTransformersDataset(Dataset):
    def __init__(
        self,
        root,
        dataset_hub_path,
        dataset_name,
        text_col="text",
        label_col="label",
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
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

        texts = news_dataset[text_col]
        self.data = torch.from_numpy(embedding_model.encode(texts))
        self.targets = torch.tensor(news_dataset[label_col])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            data = self.transform(data.numpy().reshape(1, *data.shape)).reshape(
                *data.shape
            )
        if self.target_transform is not None:
            target = self.transform(target)

        return data.numpy(), target


class TwentyNews_PU(PUDatasetBase, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(20),
            positive_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="SetFit/20_newsgroups",
            dataset_name="20news",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class IMDB_PU(PUDatasetBase, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="imdb",
            dataset_name="IMDB",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class HateSpeech_PU(PUDatasetBase, DatasetSplitterMixin, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="hate_speech18",
            dataset_name="HateSpeech",
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.train = train
        if self.train:
            split_name = "train"
        else:
            split_name = "test"

        idx = self.get_split_idx(self.data, split_name, random_seed=random_seed)
        self.data = self.data[idx]
        self.targets = self.targets[idx]

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class SMSSpam_PU(PUDatasetBase, DatasetSplitterMixin, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="sms_spam",
            dataset_name="SMSSpam",
            text_col="sms",
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.train = train
        if self.train:
            split_name = "train"
        else:
            split_name = "test"

        idx = self.get_split_idx(self.data, split_name, random_seed=random_seed)
        self.data = self.data[idx]
        self.targets = self.targets[idx]

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class PoemSentiment_PU(PUDatasetBase, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(4),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="poem_sentiment",
            dataset_name="PoemSentiment",
            text_col="verse_text",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


# class SyntheticDataset(Dataset):
#     def __init__(
#         self,
#         root,
#         train=True,
#         transform=None,
#         target_transform=None,
#         download=True,  # ignored
#         size=1000,
#         random_seed=42,
#     ):
#         self.transform = transform
#         self.target_transform = target_transform

#         if not train:
#             random_seed = random_seed + 42
#         generator = torch.Generator().manual_seed(random_seed)

#         n_pos = int(0.8 * size)
#         X_pos = torch.randn((n_pos, 2), generator=generator) + torch.tensor([[2, 0]])
#         X_neg = torch.randn((size - n_pos, 2), generator=generator) + torch.tensor(
#             [[-2, 0]]
#         )

#         X = torch.cat([X_pos, X_neg])
#         y = torch.cat([torch.ones(len(X_pos)), torch.zeros(len(X_neg))])

#         self.data, self.targets = X, y

#     def __len__(self):
#         # if self.train:
#         return len(self.targets)

#     def __getitem__(self, idx):
#         # return super().__getitem__(idx)
#         data, target = self.data[idx], self.targets[idx]

#         if self.transform is not None:
#             data = self.transform(data.numpy().reshape(1, *data.shape)).reshape(
#                 *data.shape
#             )
#         if self.target_transform is not None:
#             target = self.transform(target)

#         return data.numpy(), target


class TabularBenchmarkDataset(DatasetSplitterMixin, Dataset):
    def __init__(
        self,
        root,
        dataset_name,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        dataset = load_dataset(
            "polinaeterna/tabular-benchmark",
            data_files=f"clf_num/{dataset_name}.csv",
            cache_dir=os.path.join(root, dataset_name),
        )
        dataset = dataset["train"]

        data = dataset.remove_columns(
            [dataset.column_names[0], dataset.column_names[-1]]
        )
        targets = dataset.select_columns(dataset.column_names[-1])

        X = pd.DataFrame(data).values
        y = pd.DataFrame(targets).values.reshape(-1)

        X = StandardScaler().fit_transform(X)

        self.train = train
        if self.train:
            split_name = "train"
        else:
            split_name = "test"

        idx = self.get_split_idx(dataset, split_name, random_seed=random_seed)
        X = X[idx]
        y = y[idx]

        self.data = torch.tensor(X).float()
        if y.dtype != object:
            self.targets = torch.tensor(y).int()
        else:
            self.targets = torch.from_numpy(
                np.where(y == np.unique(y)[0], 1, 0),
            )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            data = self.transform(data.numpy().reshape(1, *data.shape)).reshape(
                *data.shape
            )
        if self.target_transform is not None:
            target = self.transform(target)

        return data.numpy(), target


class TabularBenchmark_PU(PUDatasetBase, TabularBenchmarkDataset):
    def __init__(
        self,
        root,
        dataset_name,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmarkDataset.__init__(
            self,
            root=root,
            dataset_name=dataset_name,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class TBCredit_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "credit",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


class TBCalifornia_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "california",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


class TBWine_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "wine",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


class TBElectricity_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "electricity",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


class TBCovertype_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "covertype",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


# class ImageDataset(Dataset):
#     def __init__(
#         self,
#         root,
#         dataset_name,
#         train=True,
#         transform=None,
#         target_transform=None,
#         download=True,  # ignored
#         random_seed=None,
#     ):
#         dataset = load_dataset(
#             "polinaeterna/tabular-benchmark",
#             data_files=f"clf_num/{dataset_name}.csv",
#             cache_dir=os.path.join(root, dataset_name),
#         )
#         self.dataset = dataset

#         self.transform = transform
#         self.target_transform = target_transform

from torchvision.transforms import Compose, Resize, ToTensor


class Snacks_PU(PUDatasetBase):
    def __init__(
        self,
        root,
        dataset_name,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(20),
            positive_classes=[0, 1, 4, 7, 12, 13, 17, 19],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        dataset = load_dataset(
            "Matthijs/snacks",
            cache_dir=os.path.join(root, dataset_name),
        )

        self.train = train
        if self.train:
            dataset = dataset["train"]
        else:
            dataset = dataset["test"]

        def transforms(examples):
            examples["data"] = [
                np.array(
                    image.convert("RGB").resize((224, 224)),
                )
                / 255.0
                for image in examples["image"]
            ]
            return examples

        dataset = dataset.map(transforms, remove_columns=["image"], batched=True)
        self.data = dataset
        self.targets = torch.tensor(dataset["label"])

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()

    def __getitem__(self, idx):
        input = self.data[idx]["data"]
        target = self.binary_targets[idx]
        label = self.pu_targets[idx]
        return input, target, label


# class DogFood_PU(PUDatasetBase):
#     def __init__(
#         self,
#         root,
#         dataset_name,
#         pu_labeler: PULabeler,
#         target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
#             included_classes=np.arange(20),
#             positive_classes=[0, 1, 4, 7, 12, 13, 17, 19],
#         ),
#         train=True,
#         transform=None,
#         target_transform=None,
#         download=True,  # ignored
#         random_seed=None,
#     ):
#         dataset = load_dataset(
#             "lewtun/dog_food",
#             cache_dir=os.path.join(root, dataset_name),
#         )

#         self.train = train
#         if self.train:
#             dataset = dataset["train"]
#         else:
#             dataset = dataset["test"]

#         self.data = [np.array(img) / 255.0 for img in dataset["image"]]
#         self.targets = torch.tensor(dataset["label"])

#         self.target_transformer = target_transformer
#         self.pu_labeler = pu_labeler
#         self._convert_to_pu_data()


# dataset = Snacks_PU("data", "Snacks", SCAR_SS_Labeler(0.5))
# dataset
