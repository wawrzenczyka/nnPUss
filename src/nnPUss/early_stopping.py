import numpy as np
import torch
from torch import nn


class EarlyStopping:
    mode_dict = {
        "min": np.less,
        "max": np.greater,
    }

    def __init__(
        self,
        model_path: str,
        num_epochs: int,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
    ):
        super().__init__()
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait_count = 0
        self.stopped_epoch = 0
        self.mode = mode

        if mode not in self.mode_dict:
            if self.verbose > 0:
                print(f"EarlyStopping mode {mode} is unknown, fallback to min mode.")
            self.mode = "min"

        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf

    @property
    def monitor_op(self):
        return self.mode_dict[self.mode]

    def check(self, epoch, model: nn.Module, new_metric_value):
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        if self.monitor_op(new_metric_value - self.min_delta, self.best_score):
            self.best_score = new_metric_value
            self.wait_count = 0

            if self.model_path is not None:
                torch.save(model.state_dict(), self.model_path)

            should_stop = False
        else:
            self.wait_count += 1
            should_stop = bool(self.wait_count >= self.patience)

            if should_stop:
                self.stopped_epoch = epoch
                self.best_epoch = epoch - self.wait_count
                if self.verbose:
                    print(
                        f"Early stopping at epoch {epoch - self.wait_count + 1}, best score: {self.best_score}"
                    )
                if self.model_path is not None:
                    model.load_state_dict(torch.load(self.model_path))

        if epoch == self.num_epochs - 1:
            self.stopped_epoch = epoch
            self.best_epoch = epoch - self.wait_count
            print(
                f"Training finished without early stoppping, best model at epoch {epoch - self.wait_count + 1}, score: {self.best_score}"
            )
            should_stop = True

        return should_stop
