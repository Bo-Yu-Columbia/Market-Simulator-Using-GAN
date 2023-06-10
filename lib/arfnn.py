"""
This module provides the ResidualBlock, ResFNN, ArFNN, and SimpleGenerator classes. These classes 
define the structure of various types of feedforward neural networks with residual connections. 
These networks are used in the generative algorithms to model the data.

Classes:
    - ResidualBlock: Defines a single residual block in the network.
    - ResFNN: Feedforward neural network with residual connection.
    - ArFNN: Autoregressive feedforward neural network with residual connection.
    - SimpleGenerator: A simple generator network that extends the ArFNN.

"""

from typing import Tuple

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU()
        self.create_residual_connection = True if input_dim == output_dim else False

    def forward(self, x):
        """
        Computes the output of the residual block for a given input.
        """
        y = self.activation(self.linear(x))
        if self.create_residual_connection:
            y = x + y
        return y


class ResFNN(nn.Module):
    """
    Feedforward neural network with residual connection.

    Args:
        input_dim: integer, specifies the input dimension of the neural network
        output_dim: integer, specifies the output dimension of the neural network
        hidden_dims: list of integers, specifies the hidden dimensions of each layer
        flatten: bool, if True, it reshapes the input to a 1-D tensor
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], flatten: bool = False):
        """
        Feedforward neural network with residual connection.

        Args:
            input_dim: integer, specifies input dimension of the neural network
            output_dim: integer, specifies output dimension of the neural network
            hidden_dims: list of integers, specifies the hidden dimensions of each layer.
                in above definition L = len(hidden_dims) since the last hidden layer is followed by an output layer
        """
        super(ResFNN, self).__init__()
        blocks = list()
        self.input_dim = input_dim
        self.flatten = flatten
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        blocks.append(nn.Linear(input_dim_block, output_dim))
        self.network = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        """
        Computes the output of the residual block for a given input.
        """
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        out = self.network(x)
        return out


class ArFNN(nn.Module):
    """
    Autoregressive feedforward neural network with residual connection.

    Args:
        input_dim: integer, specifies the input dimension of the neural network
        output_dim: integer, specifies the output dimension of the neural network
        hidden_dims: list of integers, specifies the hidden dimensions of each layer
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int]):
        super().__init__()
        self.network = ResFNN(input_dim, output_dim, hidden_dims)

    def forward(self, z, x_past):
        """
        Computes the output of the residual block for a given input.
        """
        x_generated = list()
        for t in range(z.shape[1]):
            z_t = z[:, t:t + 1]
            x_in = torch.cat([z_t, x_past.reshape(x_past.shape[0], 1, -1)], dim=-1)
            x_gen = self.network(x_in)
            x_past = torch.cat([x_past[:, 1:], x_gen], dim=1)
            x_generated.append(x_gen)
        x_fake = torch.cat(x_generated, dim=1)
        return x_fake


class SimpleGenerator(ArFNN):
    """
    A simple generator network that extends the ArFNN. 
    It adds the capability to generate samples.

    Args:
        input_dim: integer, specifies the input dimension of the neural network
        output_dim: integer, specifies the output dimension of the neural network
        hidden_dims: list of integers, specifies the hidden dimensions of each layer
        latent_dim: integer, specifies the dimension of the latent space
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], latent_dim: int):
        super(SimpleGenerator, self).__init__(input_dim + latent_dim, output_dim, hidden_dims)
        self.latent_dim = latent_dim

    def sample(self, steps, x_past):
        """
        Generates samples from the network.

        Args:
            steps: integer, number of time steps to generate
            x_past: tensor, past values of the time series
        
        Returns:
            The generated samples.
        """
        z = torch.randn(x_past.size(0), steps, self.latent_dim).to(x_past.device)
        return self.forward(z, x_past)
