# %%
from dataset_configs import DatasetConfigs
from experiment_config import ExperimentConfig
from loss import nnPUccLoss, nnPUssLoss, uPUccLoss, uPUssLoss
from run_experiment import Experiment

if __name__ == "__main__":
    for exp_number in range(0, 1):
        # for exp_number in range(2, 4):
        # for exp_number in range(4, 6):
        # for exp_number in range(6, 8):
        # for exp_number in range(8, 10):
        for label_frequency in [
            # 0.3,
            # 0.8,
            # 0.5,
            # 0.02,
            # 0.1,
            # 0.3,
            0.7,
            # 1,
        ]:
            for dataset_config in [
                # DatasetConfigs.Synthetic_SS,
                # DatasetConfigs.Synthetic_CC,
                DatasetConfigs.MNIST_SS,
                # DatasetConfigs.MNIST_CC,
                # DatasetConfigs.MNIST_SS_joined,
                # DatasetConfigs.MNIST_CC_joined,
                # DatasetConfigs.TwentyNews_CC,
                # DatasetConfigs.TwentyNews_SS,
                # DatasetConfigs.IMDB_CC,
                # DatasetConfigs.IMDB_SS,
            ]:
                for PULoss in [
                    # nnPUssLoss,
                    nnPUccLoss,
                    # uPUssLoss,
                    # uPUccLoss,
                ]:
                    experiment_config = ExperimentConfig(
                        PULoss=PULoss,
                        dataset_config=dataset_config,
                        label_frequency=label_frequency,
                        exp_number=exp_number,
                    )
                    print(f"Starting: {experiment_config}...")

                    experiment = Experiment(experiment_config)
                    experiment.run()

                    print(f"Finished: {experiment_config}")

# %%
