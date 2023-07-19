from torchvision import transforms

from src.nnPUss.dataset import (
    IMDB_PU,
    MNIST_PU,
    HateSpeech_PU,
    PoemSentiment_PU,
    PUDatasetBase,
    PULabeler,
    SCAR_CC_Labeler,
    SCAR_SS_Labeler,
    SMSSpam_PU,
    TBCalifornia_PU,
    TBCovertype_PU,
    TBCredit_PU,
    TBElectricity_PU,
    TBWine_PU,
    TwentyNews_PU,
)


class DatasetConfig:
    def __init__(
        self,
        name: str,
        DatasetClass: type[PUDatasetBase],
        PULabelerClass: type[PULabeler],
        num_epochs=50,
        learning_rate=1e-5,
    ):
        self.name = name
        self.DatasetClass = DatasetClass
        self.PULabelerClass = PULabelerClass
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        # self.normalization = normalization


class DatasetConfigs:
    MNIST_CC = DatasetConfig(
        "[IMG] MNIST CC",
        DatasetClass=MNIST_PU,
        PULabelerClass=SCAR_CC_Labeler,
        # normalization=transforms.Normalize((0.1307,), (0.3081,)),
    )
    MNIST_SS = DatasetConfig(
        "[IMG] MNIST SS",
        DatasetClass=MNIST_PU,
        PULabelerClass=SCAR_SS_Labeler,
        # normalization=transforms.Normalize((0.1307,), (0.3081,)),
    )
    TwentyNews_CC = DatasetConfig(
        "[TXT] 20News CC",
        DatasetClass=TwentyNews_PU,
        PULabelerClass=SCAR_CC_Labeler,
        # normalization=transforms.Normalize((-0.0004,), (0.0510,)),
        learning_rate=5e-5,
        num_epochs=50,
    )
    TwentyNews_SS = DatasetConfig(
        "[TXT] 20News SS",
        DatasetClass=TwentyNews_PU,
        PULabelerClass=SCAR_SS_Labeler,
        # normalization=transforms.Normalize((-0.0004,), (0.0510,)),
        learning_rate=5e-5,
        num_epochs=50,
    )

    IMDB_CC = DatasetConfig(
        "[TXT] IMDB CC",
        DatasetClass=IMDB_PU,
        PULabelerClass=SCAR_CC_Labeler,
        # normalization=transforms.Normalize((-0.0005,), (0.0510,)),
    )
    IMDB_SS = DatasetConfig(
        "[TXT] IMDB SS",
        DatasetClass=IMDB_PU,
        PULabelerClass=SCAR_SS_Labeler,
        # normalization=transforms.Normalize((-0.0005,), (0.0510,)),
    )

    PoemSentiment_CC = DatasetConfig(
        "[TXT] PoemSentiment CC",
        DatasetClass=PoemSentiment_PU,
        PULabelerClass=SCAR_CC_Labeler,
        learning_rate=5e-5,
        num_epochs=150,
    )
    PoemSentiment_SS = DatasetConfig(
        "[TXT] PoemSentiment SS",
        DatasetClass=PoemSentiment_PU,
        PULabelerClass=SCAR_SS_Labeler,
        learning_rate=5e-5,
        num_epochs=150,
    )

    HateSpeech_CC = DatasetConfig(
        "[TXT] HateSpeech CC",
        DatasetClass=HateSpeech_PU,
        PULabelerClass=SCAR_CC_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )
    HateSpeech_SS = DatasetConfig(
        "[TXT] HateSpeech SS",
        DatasetClass=HateSpeech_PU,
        PULabelerClass=SCAR_SS_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )

    SMSSpam_CC = DatasetConfig(
        "[TXT] SMSSpam CC",
        DatasetClass=SMSSpam_PU,
        PULabelerClass=SCAR_CC_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )
    SMSSpam_SS = DatasetConfig(
        "[TXT] SMSSpam SS",
        DatasetClass=SMSSpam_PU,
        PULabelerClass=SCAR_SS_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )

    TB_Credit_CC = DatasetConfig(
        "[TAB] Credit CC",
        DatasetClass=TBCredit_PU,
        PULabelerClass=SCAR_CC_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )
    TB_Credit_SS = DatasetConfig(
        "[TAB] Credit SS",
        DatasetClass=TBCredit_PU,
        PULabelerClass=SCAR_SS_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )

    TB_California_CC = DatasetConfig(
        "[TAB] California CC",
        DatasetClass=TBCalifornia_PU,
        PULabelerClass=SCAR_CC_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )
    TB_California_SS = DatasetConfig(
        "[TAB] California SS",
        DatasetClass=TBCalifornia_PU,
        PULabelerClass=SCAR_SS_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )

    TB_Wine_CC = DatasetConfig(
        "[TAB] Wine CC",
        DatasetClass=TBWine_PU,
        PULabelerClass=SCAR_CC_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )
    TB_Wine_SS = DatasetConfig(
        "[TAB] Wine SS",
        DatasetClass=TBWine_PU,
        PULabelerClass=SCAR_SS_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )

    TB_Electricity_CC = DatasetConfig(
        "[TAB] Electricity CC",
        DatasetClass=TBElectricity_PU,
        PULabelerClass=SCAR_CC_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )
    TB_Electricity_SS = DatasetConfig(
        "[TAB] Electricity SS",
        DatasetClass=TBElectricity_PU,
        PULabelerClass=SCAR_SS_Labeler,
        num_epochs=100,
        learning_rate=5e-5,
    )

    # Synthetic_SS = DatasetConfig(
    #     "Synthetic SS",
    #     Synthetic_PU_SS,
    #     positive_labels=[1],
    #     normalization=None,
    # )
    # Synthetic_CC = DatasetConfig(
    #     "Synthetic CC",
    #     Synthetic_PU_CC,
    #     positive_labels=[1],
    #     normalization=None,
    # )
