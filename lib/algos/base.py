"""
base.py

This module provides the base classes and methods that are used by the training script (train.py) 
for running various generative algorithms. The main components are the BaseConfig data class 
which holds the configuration parameters for the algorithms, the BaseAlgo class which provides 
the main structure of the generative algorithms, and several utility functions.

Classes:
    - BaseConfig: Holds the configuration for the generative algorithms.
    - BaseAlgo: Provides the main structure for the generative algorithms.

Functions:
    - is_multivariate(x): Checks if the input is multivariate.
    - get_standard_test_metrics(x): Returns a list of test metrics for the input.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from lib.arfnn import SimpleGenerator
from lib.test_metrics import test_metrics
from lib.utils import to_numpy


@dataclass
class BaseConfig:
    """
    A data class that holds the configuration parameters for the generative algorithms.

    Attributes:
        seed (int): Seed for reproducibility. Default is 0.
        batch_size (int): Size of the batches for training. Default is 200.
        device (str): The device to run the training on (cpu/gpu). Default is 'cpu'.
        p (int): The lag order for the autoregressive model. Default is 3.
        q (int): The lag order for the moving average model. Default is 3.
        hidden_dims (Tuple[int]): The hidden layer dimensions. Default is 3 layers of 50 neurons each.
        total_steps (int): The total number of training steps. Default is 1000.
        mc_samples (int): The number of Monte Carlo samples. Default is 1000.
    """

    seed: int = 0
    batch_size: int = 200
    device: str = 'cpu'
    p: int = 3
    q: int = 3
    hidden_dims: Tuple[int] = 3 * (50,)
    total_steps: int = 1000
    mc_samples: int = 1000


def is_multivariate(x):
    """
    Checks if the input is multivariate. 

    Parameters:
        x: The input to check.

    Returns:
        bool: True if the input is multivariate, False otherwise.
    """
    return True if x.shape[-1] > 1 else False


def get_standard_test_metrics(x):
    """
    Returns a list of test metrics for the input.

    Parameters:
        x: The input to generate test metrics for.

    Returns:
        list: A list of test metrics for the input.
    """
    test_metrics_list = [
        test_metrics['abs_metric'](x, reg=0.1),
        test_metrics['acf_id'](x, max_lag=2, reg=0.3),
    ]
    if is_multivariate(x):
        test_metrics_list.append(test_metrics['cross_correl'](x, reg=0.1))
    return test_metrics_list


class BaseAlgo:
    """
    A class that provides the main structure for the generative algorithms.

    Attributes:
        base_config (BaseConfig): The configuration for the algorithm.
        batch_size (int): The batch size for training.
        hidden_dims (Tuple[int]): The hidden layer dimensions.
        p, q (int): The lag orders for the autoregressive and moving average models respectively.
        total_steps (int): The total number of training steps.
        device (str): The device to run the training on.
        x_real: The real data to be generated.
        dim, latent_dim (int): The dimensions of the real data and the latent space respectively.
        training_loss (defaultdict): A dictionary to keep track of the training losses.
        test_metrics_list (list): A list of test metrics for the real data.
        G (SimpleGenerator): The generator model.

    Methods:
        fit(): Runs the training process.
        step(): Defines a single step in the training process.
        evaluate(x_fake): Evaluates the model using fake data.
        plot_losses(): Plots the training and test metric losses.
    """

    def __init__(self, base_config, x_real):
        self.base_config = base_config
        self.batch_size = base_config.batch_size
        self.hidden_dims = base_config.hidden_dims
        self.p, self.q = base_config.p, base_config.q
        self.total_steps = base_config.total_steps

        self.device = base_config.device
        self.x_real = x_real
        self.dim = self.latent_dim = x_real.shape[-1]

        self.training_loss = defaultdict(list)
        self.test_metrics_list = get_standard_test_metrics(x_real)

        self.G = SimpleGenerator(self.p * self.dim, self.dim, self.hidden_dims, self.latent_dim).to(self.device)

    def fit(self):
        """
        Runs the training process. The batch size is set to the size of the real data if it is -1. 
        The training process is run for the number of steps defined in the base configuration.
        """
        if self.batch_size == -1:
            self.batch_size = self.x_real.shape[0]
        for _ in tqdm(range(self.total_steps), ncols=80):  # sig_pred, x_past, x_real
            self.step()

    def step(self):
        """
        Defines a single step in the training process. This is an abstract method that needs to 
        be implemented by the child class.
        """
        raise NotImplementedError('Needs implementation by child.')

    def evaluate(self, x_fake):
        """
        Evaluates the model using fake data.

        Parameters:
            x_fake: The fake data to use for the evaluation.
        """
        for test_metric in self.test_metrics_list:
            with torch.no_grad():
                test_metric(x_fake[:10000])
            self.training_loss[test_metric.name].append(
                to_numpy(test_metric.loss_componentwise)
            )

    def plot_losses(self):
        """
        Plots the training and test metric losses.
        """
        fig, axes = plt.subplots(len(self.test_metrics_list) + 1, 1, figsize=(10, 8))
        algo = type(self).__name__
        if algo in ['RCGAN', 'TimeGAN', 'RCWGAN']:
            axes[0].plot(self.training_loss['G_loss'], label='Generator loss')
            if algo in ['RCGAN', 'TimeGAN']:
                axes[0].plot(self.training_loss['D_loss'], label='Discriminator loss')
            elif algo == 'RCWGAN':
                axes[0].plot(self.training_loss['D_loss'], label='Critic loss')
                axes[0].plot(self.training_loss['RCWGAN_reg'], label='GP')
        elif algo == 'GMMN':
            axes[0].plot(self.training_loss['MMD'], label='MMD')
        elif algo == 'SigCWGAN':
            loss = self.training_loss['loss']
            axes[0].plot(loss, label='Sig-$W_1$ loss')
            axes[0].set_yscale('log')
        else:
            raise NotImplementedError('Algo "%s" not implemented' % algo)
        axes[0].grid()
        axes[0].legend()
        for i, test_metric in enumerate(self.test_metrics_list):
            axes[i + 1].plot(self.training_loss[test_metric.name], label=test_metric.name)
            axes[i + 1].grid()
            axes[i + 1].legend()
            axes[i + 1].set_ylim(bottom=0.)
            if i + 1 == len(self.test_metrics_list):
                axes[i + 1].set_xlabel('Generator weight updates')
