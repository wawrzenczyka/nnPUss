from attr import dataclass


@dataclass
class MetricValues:
    model: str
    dataset: str
    label_frequency: float
    exp_number: int  # seed

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float

    loss: float
    stopping_epoch: int
    time: float
