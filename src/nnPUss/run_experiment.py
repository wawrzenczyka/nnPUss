# %%
import json
import os
import time
from typing import Type

import mlflow
import numpy as np
import pkbar
import torch
from experiment_config import ExperimentConfig
from loss import _PULoss
from metric_values import MetricValues
from model import PUModel
from sklearn import metrics
from torch.optim import Adam
from torch.utils.data import DataLoader


class DictJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Type):
            return o.__module__ + "." + o.__class__.__name__
        return o.__dict__


class Experiment:
    def __init__(self, experiment_config: ExperimentConfig) -> None:
        self.experiment_config = experiment_config
        self.prepare_experiment(experiment_config)

        use_cuda = not self.experiment_config.force_cpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._prepare_data()

        self._set_seed()

        self.model = PUModel(self.n_inputs)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.experiment_config.dataset_config.learning_rate,
            weight_decay=0.005,
        )

    def prepare_experiment(self, experiment_config: ExperimentConfig):
        experiment_name = experiment_config.dataset_config.name

        experiments = mlflow.search_experiments(
            filter_string=f"name = '{experiment_name}'", max_results=1
        )
        if len(experiments) == 0:
            id = mlflow.create_experiment(experiment_name)
            self.experiment = mlflow.get_experiment(id)
        else:
            self.experiment = experiments[0]

    def run(self):
        with mlflow.start_run(experiment_id=self.experiment.experiment_id):
            mlflow.log_params(
                json.loads(json.dumps(self.experiment_config, cls=DictJsonEncoder))
            )
            os.makedirs(self.experiment_config.output_dir, exist_ok=True)

            self._set_seed()

            self.model = self.model.to(self.device)
            training_start_time = time.perf_counter()
            for epoch in range(self.experiment_config.dataset_config.num_epochs):
                kbar = pkbar.Kbar(
                    target=len(self.train_loader) + 1,
                    epoch=epoch,
                    num_epochs=self.experiment_config.dataset_config.num_epochs,
                    width=8,
                    always_stateful=False,
                )

                self._train(epoch, kbar)
                self._test(epoch, kbar)

            self.training_time = time.perf_counter() - training_start_time

            kbar = pkbar.Kbar(
                target=1,
                epoch=epoch,
                num_epochs=self.experiment_config.dataset_config.num_epochs,
                width=8,
                always_stateful=False,
            )
            self._test(epoch, kbar, save_metrics=True)

    def _prepare_data(self):
        self._set_seed()

        data = {}
        for is_train_set in [True, False]:
            data[
                "train" if is_train_set else "test"
            ] = self.experiment_config.dataset_config.DatasetClass(
                self.experiment_config.data_dir,
                self.experiment_config.dataset_config.PULabelerClass(
                    label_frequency=self.experiment_config.label_frequency
                ),
                train=is_train_set,
                download=True,
                random_seed=self.experiment_config.seed,
            )

        train_set = data["train"]
        self.prior = train_set.get_prior()
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.experiment_config.dataset_config.train_batch_size,
            shuffle=True,
        )
        self.n_inputs = len(next(iter(train_set))[0].reshape(-1))

        test_set = data["test"]
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.experiment_config.dataset_config.eval_batch_size,
            shuffle=False,
        )

    def _train(self, epoch: int, kbar: pkbar.Kbar):
        self.model.train()
        tr_loss = 0

        loss_fct = self.experiment_config.PULoss(prior=self.prior)

        for batch_idx, (data, target, label) in enumerate(self.train_loader):
            data, label = data.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = loss_fct(output.view(-1), label.type(torch.float))
            tr_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            y_pred = torch.where(output < 0, -1, 1).to(self.device)
            acc = metrics.accuracy_score(
                target.cpu().numpy().reshape(-1), y_pred.cpu().numpy().reshape(-1)
            )
            kbar.update(
                batch_idx + 1, values=[("train_loss", loss), ("train_acc", acc)]
            )

        train_metrics = {k: np.mean(v) for k, v in kbar._values.items()}
        loss_components = {k: np.mean(v) for k, v in loss_fct.history.items()}

        mlflow.log_metrics(train_metrics, step=epoch)
        mlflow.log_metrics(loss_components, step=epoch)

    def _test(self, epoch: int, kbar: pkbar.Kbar, save_metrics: bool = False):
        """Testing"""
        self.model.eval()
        test_loss = 0
        correct = 0
        num_pos = 0

        test_points = []
        targets = []
        preds = []

        test_loss_func = self.experiment_config.PULoss(prior=self.prior)

        with torch.no_grad():
            for data, target, _ in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                test_loss += test_loss_func(
                    output.view(-1), target.type(torch.float)
                ).item()  # sum up batch loss
                pred = torch.where(
                    output < 0,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )
                num_pos += torch.sum(pred == 1)
                correct += pred.eq(target.view_as(pred)).sum().item()

                test_points.append(data)
                targets.append(target)
                preds.append(pred)

        test_loss /= len(self.test_loader)

        kbar.add(
            1,
            values=[
                ("test_loss", test_loss),
                ("test_accuracy", 100.0 * correct / len(self.test_loader.dataset)),
                ("pos_fraction", float(num_pos) / len(self.test_loader.dataset)),
            ],
        )

        targets = torch.cat(targets).detach().cpu().numpy()
        preds = torch.cat(preds).detach().cpu().numpy()

        metric_values = self._calculate_metrics(test_loss, targets, preds)

        if save_metrics:
            numeric_metrics = {
                k: np.mean(v)
                for k, v in json.loads(
                    json.dumps(metric_values, cls=DictJsonEncoder)
                ).items()
                if type(v) != str
            }
            mlflow.log_metrics(numeric_metrics)

            with open(self.experiment_config.metrics_file, "w") as f:
                json.dump(metric_values, f, cls=DictJsonEncoder)

        return test_loss

    def _calculate_metrics(self, test_loss, targets, preds):
        y_true = np.where(targets == 1, 1, 0)
        y_pred = np.where(preds == 1, 1, 0)

        metric_values = MetricValues(
            model=self.experiment_config.PULoss.name,
            dataset=self.experiment_config.dataset_config.name,
            label_frequency=self.experiment_config.label_frequency,
            exp_number=self.experiment_config.exp_number,
            accuracy=metrics.accuracy_score(y_true, y_pred),
            precision=metrics.precision_score(y_true, y_pred),
            recall=metrics.precision_score(y_true, y_pred),
            f1=metrics.f1_score(y_true, y_pred),
            auc=metrics.roc_auc_score(y_true, y_pred),
            loss=test_loss,
            # stopping_epoch=self.early_stopping_epoch,
            stopping_epoch=self.experiment_config.dataset_config.num_epochs,
            time=self.training_time if hasattr(self, "training_time") else None,
        )

        return metric_values

    def _set_seed(self):
        torch.manual_seed(self.experiment_config.seed)
        np.random.seed(self.experiment_config.seed)
