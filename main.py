# %%
from dataset_configs import DatasetConfigs
from experiment_config import ExperimentConfig
from loss import nnPUccLoss, nnPUssLoss, uPUccLoss, uPUssLoss
from run_experiment import Experiment

if __name__ == "__main__":
    for exp_number in range(10):
        for label_frequency in [
            0.02,
            0.5,
            0.1,
            0.3,
            0.7,
        ]:
            for dataset_config in [
                DatasetConfigs.MNIST_CC,
                DatasetConfigs.MNIST_SS,
                DatasetConfigs.TwentyNews_CC,
                DatasetConfigs.TwentyNews_SS,
                DatasetConfigs.IMDB_CC,
                DatasetConfigs.IMDB_SS,
            ]:
                for PULoss in [nnPUccLoss, nnPUssLoss, uPUssLoss, uPUccLoss]:
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
