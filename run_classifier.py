# %%
import argparse
import logging
import multiprocessing
import os

import numpy as np
import pkbar
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchvision import transforms

from dataset import (
    MNIST_PU_CC,
    MNIST_PU_SS,
    SCARLabeler,
    TwentyNews_PU_CC,
    TwentyNews_PU_SS,
)
from dataset_configs import DatasetConfigs
from loss import _PULoss, nnPUccLoss, nnPUssLoss, uPUccLoss, uPUssLoss
from model import PUModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# label_frequency = 0.02
label_frequency = 0.5

dataset_config = DatasetConfigs.MNIST_SS
PULoss = nnPUccLoss
# dataset_config = DatasetConfigs.MNIST_SS
# PULoss = nnPUssLoss
# dataset_config = DatasetConfigs.MNIST_CC
# PULoss = nnPUccLoss
# dataset_config = DatasetConfigs.MNIST_CC
# PULoss = nnPUssLoss

# dataset_config = DatasetConfigs.TwentyNews_SS
# PULoss = nnPUccLoss
# dataset_config = DatasetConfigs.TwentyNews_SS
# PULoss = nnPUssLoss
# dataset_config = DatasetConfigs.TwentyNews_CC
# PULoss = nnPUccLoss
# dataset_config = DatasetConfigs.TwentyNews_CC
# PULoss = nnPUssLoss

seed = 1

torch.manual_seed(seed)
np.random.seed(seed)


def train(args, model, device, train_loader, optimizer, prior, epoch, kbar):
    model.train()
    tr_loss = 0

    for batch_idx, (data, _, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss_fct = PULoss(prior=prior)

        loss = loss_fct(output.view(-1), label.type(torch.float))
        tr_loss += loss.item()
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print(
        #         "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #             epoch,
        #             batch_idx * len(data),
        #             len(train_loader.dataset),
        #             100.0 * batch_idx / len(train_loader),
        #             loss.item(),
        #         )
        #     )
        kbar.update(batch_idx, values=[("loss", loss)])

    # print("Train loss: ", tr_loss)


def test(args, model, device, test_loader, prior, kbar):
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0
    num_pos = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss_func = _PULoss(prior=prior)
            test_loss += test_loss_func(
                output.view(-1), target.type(torch.float)
            ).item()  # sum up batch loss
            pred = torch.where(
                output < 0,
                torch.tensor(-1, device=device),
                torch.tensor(1, device=device),
            )
            num_pos += torch.sum(pred == 1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    kbar.add(
        1,
        values=[
            ("test_loss", test_loss),
            ("test_accuracy", 100.0 * correct / len(test_loader.dataset)),
            ("pos_fraction", float(num_pos) / len(test_loader.dataset)),
        ],
    )
    # print(
    #     "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
    #         test_loss,
    #         correct,
    #         len(test_loader.dataset),
    #         100.0 * correct / len(test_loader.dataset),
    #     )
    # )
    # print(
    #     "Percent of examples predicted positive: ",
    #     float(num_pos) / len(test_loader.dataset),
    #     "\n",
    # )


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--nnPU",
        action="store_true",
        help="Whether to us non-negative pu-learning risk estimator.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=30000,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=100, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=50,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    args = parser.parse_args(
        (
            "--data_dir ./data --output_dir ./output_2 "
            "--do_train --do_eval --nnPU "
            "--train_batch_size 30000 --eval_batch_size=100"
        ).split(" ")
    )

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if (
    #     os.path.exists(args.output_dir)
    #     and os.listdir(args.output_dir)
    #     and args.do_train
    # ):
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty.".format(
    #             args.output_dir
    #         )
    #     )
    os.makedirs(args.output_dir, exist_ok=True)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = (
        {"num_workers": min(multiprocessing.cpu_count() // 2, 1), "pin_memory": True}
        if use_cuda
        else {}
    )

    kwargs = {}

    data = {}
    for is_train_set in [True, False]:
        data["train" if is_train_set else "test"] = dataset_config.DatasetClass(
            args.data_dir,
            SCARLabeler(
                positive_labels=dataset_config.positive_labels,
                label_frequency=label_frequency,
                NEGATIVE_LABEL=-1,
            ),
            train=is_train_set,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    dataset_config.normalization,
                ]
            ),
        )

    train_set = data["train"]
    prior = train_set.get_prior()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.train_batch_size, shuffle=True, **kwargs
    )

    test_set = data["test"]

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.eval_batch_size, shuffle=False, **kwargs
    )

    n_inputs = len(next(iter(train_set))[0].reshape(-1))
    model = PUModel(n_inputs).to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.005)

    if args.do_train:
        for epoch in range(1, args.num_train_epochs + 1):
            kbar = pkbar.Kbar(
                target=len(train_loader),
                epoch=epoch - 1,
                num_epochs=args.num_train_epochs,
                width=8,
                always_stateful=False,
            )
            train(args, model, device, train_loader, optimizer, prior, epoch, kbar)
            test(args, model, device, test_loader, prior, kbar)

        torch.save(model.state_dict(), output_model_file)

    elif args.do_eval:
        test(args, model, device, test_loader, prior)


if __name__ == "__main__":
    main()

# %%
