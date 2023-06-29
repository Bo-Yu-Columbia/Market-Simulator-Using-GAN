from lib.algos.sigcwgan import SigCWGANConfig
from lib.augmentations import get_standard_augmentation, SignatureConfig, Scale, Concat, Cumsum, AddLags, LeadLag

# In this dictionary each key-value pair corresponds to a different dataset (e.g., 'ECG', 'VAR1', 'STOCKS_SPX', etc.). 
# Each dataset is associated with a SigCWGANConfig instance, which specifies the configuration of the Signature 
# Conditional Wasserstein Generative Adversarial Network (SigCWGAN) model used for that dataset

# For each additional dataset you add, you will need to add one instance to the dictionary

# Here are the parameters you need to consider:
    # mc_size: This parameter determines the Monte Carlo sample size used in SigCWGAN. It affects the variance of the gradient 
        # estimator. A larger mc_size leads to a more accurate gradient estimate but at the cost of increased computational complexity. 
        # Therefore, it's a trade-off between accuracy and efficiency. If the model is underfitting or the gradient estimate is noisy, 
        # increasing mc_size may help.
    
    # sig_config_past, sig_config_future: These are instances of the SignatureConfig class, which specify the configuration for 
        # computing the path signature of the past and future paths, respectively. 
    
    # Each SignatureConfig has the following parameters:
        # depth: This is the truncation level of the signature. A higher depth captures more complex interactions between the 
            # path components but at the cost of increased computational complexity and potential overfitting. The choice of depth should 
            # depend on the complexity of the data. If the model is underfitting, increasing depth may help.
        
        # augmentations: This is a sequence of data augmentations to apply before computing the path signature. The choice of 
            # augmentations can significantly affect the model's performance.

# There are five augmentation types:
    # Scale: This augmentation scales the data by a given factor. Scaling can help if the data values are too large or too small.

    # Cumsum: This augmentation applies the cumulative sum function to the data. It can help to capture trends in the data.

    # AddLags: This augmentation adds lagged versions of the data to itself. It can help to capture temporal dependencies in the data.

    # LeadLag: This augmentation applies the lead-lag transformation to the data. It can help to capture both lead and lag dependencies in the data.

    # Concat: This augmentation concatenates the data with itself. It can help when the data has a complex structure that can't be captured by a single path.

SIGCWGAN_CONFIGS = dict(
    ECG=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.05)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.05)),
    ),
    VAR1=SigCWGANConfig(
        mc_size=500,
        sig_config_past=SignatureConfig(depth=3, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=3,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    VAR2=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    VAR3=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=2, augmentations=get_standard_augmentation(0.2)),
    ),
    VAR20=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=(Scale(0.5), Cumsum(), Concat())),
        sig_config_future=SignatureConfig(depth=2, augmentations=(Scale(0.5), Cumsum(), Concat())),
    ),
    STOCKS_SPX=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=3, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=3,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    STOCKS_SPX_DJI=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    YIELD_1Mo=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    YIELD_1Mo_3Mo=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    YIELD_1Mo_3Mo_1Yr=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    YIELD_1Yr=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    YIELD_1Yr_3Yr=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    YIELD_1Yr_3Yr_10Yr=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    EIB_1yr=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    EXCHANGE_JPYUSD=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    EXCHANGE_JPYUSD_EURUSD=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    ARCH=SigCWGANConfig(
        mc_size=2000,
        sig_config_past=SignatureConfig(depth=3, augmentations=get_standard_augmentation(0.2)),
        sig_config_future=SignatureConfig(depth=3, augmentations=get_standard_augmentation(0.2)),
    ),
)
