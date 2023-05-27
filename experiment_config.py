import os
from typing import Any, Type

from attr import dataclass

from dataset_configs import DatasetConfig
from loss import _PULoss


@dataclass
class ExperimentConfig:
    PULoss: Type[_PULoss]
    dataset_config: DatasetConfig
    label_frequency: float
    exp_number: int  # seed

    data_dir: str = os.path.join("data")
    output_root_dir: str = os.path.join("output")

    train_batch_size: int = 512
    eval_batch_size: int = 128

    learning_rate: float = 1e-4
    num_epochs: int = 100

    force_cpu: bool = False
    # force_cpu: bool = True

    @property
    def seed(self):
        return 42 + self.exp_number

    @property
    def output_dir(self):
        return os.path.join(
            self.output_root_dir,
            self.dataset_config.name,
            self.PULoss.name,
            str(self.label_frequency),
            str(self.exp_number),
        )

    @property
    def metrics_file(self):
        return os.path.join(self.output_dir, "metrics.json")

    @property
    def model_file(self):
        return os.path.join(self.output_dir, "pytorch_model.bin")

    def __str__(self) -> str:
        return f"{self.dataset_config.name}, c={self.label_frequency}, exp {self.exp_number}; {self.PULoss.name}"
