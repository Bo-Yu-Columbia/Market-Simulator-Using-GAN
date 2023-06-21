# This script uses PyTorch to implement the Signature-based Conditional Wasserstein Generative Adversarial Network (SigCWGAN). 
# GANs consist of two neural networks, a generator and a discriminator, trained together. The generator produces artificial 
# outputs (samples), and the discriminator tries to distinguish these artificial samples from real data.

# Key parameters and concepts include:

    # mc_size: The Monte Carlo sample size, which specifies the number of samples to be drawn in each iteration of the training process.
    # sig_config_future: This configuration object is used to compute the signatures of the future paths.
    # sig_config_past: This configuration object is used to compute the signatures of the past paths.
    # x_real: The real data against which the GAN's performance will be compared.

from dataclasses import dataclass

import torch
from sklearn.linear_model import LinearRegression
from torch import optim

from lib.algos.base import BaseAlgo, BaseConfig
from lib.augmentations import SignatureConfig
from lib.augmentations import augment_path_and_compute_signatures
from lib.utils import sample_indices, to_numpy


def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    # This function defines the loss function for the SigCWGAN.
    # The loss is calculated as the mean 2-norm of the difference between the predicted and the actual signature.
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()

def sigcwgan_loss_new(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    # Calculate the Wasserstein distance between the predicted and actual signatures
    return torch.mean(sig_pred - sig_fake_conditional_expectation)


def congan_loss(D_real: torch.Tensor, D_fake: torch.Tensor):
    # Calculate the adversarial loss for Congan
    loss_real = torch.mean(torch.log(D_real))
    loss_fake = torch.mean(torch.log(1 - D_fake))
    loss = -loss_real - loss_fake
    return loss

def hinge_gan_loss(D_real: torch.Tensor, D_fake: torch.Tensor):
    # Calculate the hinge GAN loss
    loss_real = torch.mean(torch.nn.ReLU()(1 - D_real))
    loss_fake = torch.mean(torch.nn.ReLU()(1 + D_fake))
    loss = loss_real + loss_fake
    return loss

def mse_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    # Calculate the mean squared error loss
    return torch.mean((sig_pred - sig_fake_conditional_expectation) ** 2)

def mae_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    # Calculate the mean absolute error loss
    return torch.mean(torch.abs(sig_pred - sig_fake_conditional_expectation))

def kld_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    # Calculate the Kullback-Leibler divergence loss
    loss = torch.kl_div(torch.log(sig_pred), sig_fake_conditional_expectation, reduction='batchmean')
    return loss

def bce_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    # Calculate the binary cross entropy loss
    loss = torch.nn.BCELoss()
    return loss(sig_pred, sig_fake_conditional_expectation)



@dataclass
class SigCWGANConfig:
    # This class defines the configuration for the SigCWGAN. It includes methods to compute signatures for past and future data.

    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig

    def compute_sig_past(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_past)

    def compute_sig_future(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_future)


def calibrate_sigw1_metric(config, x_future, x_past):
    # This function calibrates the Wasserstein-1 metric using linear regression.

    sigs_past = config.compute_sig_past(x_past)
    sigs_future = config.compute_sig_future(x_future)
    assert sigs_past.size(0) == sigs_future.size(0)
    X, Y = to_numpy(sigs_past), to_numpy(sigs_future)
    lm = LinearRegression()
    lm.fit(X, Y)
    sigs_pred = torch.from_numpy(lm.predict(X)).float().to(x_future.device)
    return sigs_pred


def sample_sig_fake(G, q, sig_config, x_past):
    # This function samples fake signatures and fake data from the generator.

    x_past_mc = x_past.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    x_fake = G.sample(q, x_past_mc)
    sigs_fake_future = sig_config.compute_sig_future(x_fake)
    sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past.size(0), -1).mean(0)
    return sigs_fake_ce, x_fake


# The SigCWGAN class extends BaseAlgo class. It holds the model's configurations and the real data samples.
class SigCWGAN(BaseAlgo):
    
    # The constructor initializes an instance of the SigCWGAN model.
    # It assigns the passed configurations and real data samples to the instance variables.
    # It also sets up an optimizer and a learning rate scheduler for the generator.
    def __init__(
            self,
            base_config: BaseConfig,  # Basic configuration for the base GAN algorithm
            config: SigCWGANConfig,  # Configuration specific for the SigCWGAN model
            x_real: torch.Tensor,  # Real data samples
            loss_fn  # Loss function
    ):
        super(SigCWGAN, self).__init__(base_config, x_real)  # Initialize the base GAN with base_config and real data
        self.sig_config = config  # Assign passed SigCWGAN configuration to the instance variable
        self.mc_size = config.mc_size  # Assign Monte Carlo size from the configuration to the instance variable

        # Split the real data into past and future parts
        self.x_past = x_real[:, :self.p]
        x_future = x_real[:, self.p:]
        
        # Compute signatures for the past and future parts
        self.sigs_pred = calibrate_sigw1_metric(config, x_future, self.x_past)

        # Set up an optimizer and a learning rate scheduler for the generator
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-2)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)
        self.loss_fn = loss_fn

    # This method randomly samples a batch of signatures and corresponding past data points.
    # These samples are cloned and moved to the device (GPU/CPU) where the model is running.
    def sample_batch(self, ):
        random_indices = sample_indices(self.sigs_pred.shape[0], self.batch_size)  # Randomly select batch_size number of indices
        sigs_pred = self.sigs_pred[random_indices].clone().to(self.device)  # Extract and clone the signatures at the selected indices, and move them to the device
        x_past = self.x_past[random_indices].clone().to(self.device)  # Extract and clone the past data points at the selected indices, and move them to the device
        return sigs_pred, x_past

    # This method performs a single training step for the generator.
    # It computes the loss, backpropagates the gradients, and updates the generator's parameters.
    def step(self):
        self.G.train()  # Set the generator to the training mode
        self.G_optimizer.zero_grad()  # Clear the gradients from the previous step

        sigs_pred, x_past = self.sample_batch()  # Sample a batch of signatures and past data points

        # Generate fake signatures and fake data points using the generator
        sigs_fake_ce, x_fake = sample_sig_fake(self.G, self.q, self.sig_config, x_past)
        if self.loss_fn == 1:
            loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)

        elif self.loss_fn == 2:
            loss = sigcwgan_loss_new(sigs_pred, sigs_fake_ce)

        elif self.loss_fn == 3:
            loss = congan_loss(sigs_pred, sigs_fake_ce)

        elif self.loss_fn == 4:
            loss = hinge_gan_loss(sigs_pred, sigs_fake_ce)

        elif self.loss_fn == 5:
            loss = mse_loss(sigs_pred, sigs_fake_ce)

        elif self.loss_fn == 6:
            loss = mae_loss(sigs_pred, sigs_fake_ce)

        elif self.loss_fn == 7:
            loss = kld_loss(sigs_pred, sigs_fake_ce)

        elif self.loss_fn == 8:
            loss = bce_loss(sigs_pred, sigs_fake_ce)
        # Compute the loss between the real and fake signatures
        # loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)
        # loss = sigcwgan_loss_new(sigs_pred, sigs_fake_ce)
        # loss = congan_loss(sigs_pred, sigs_fake_ce)
        # loss = hinge_gan_loss(sigs_pred, sigs_fake_ce)
        # loss = self.loss_fn(sigs_pred, sigs_fake_ce)


        loss.backward()  # Compute the gradients by backpropagation

        # Clip the gradients to prevent them from blowing up
        total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)

        # Store the loss and the total norm of gradients for monitoring
        self.training_loss['loss'].append(loss.item())
        self.training_loss['total_norm'].append(total_norm)

        # Update the generator's parameters according to the computed gradients
        self.G_optimizer.step()  

        # Update the learning rate according to the scheduler
        self.G_scheduler.step()

        # Evaluate the performance of the model using the generated fake data
        self.evaluate(x_fake)

# 'evaluate' method is not included in the provided code, but it's used in GANs to assess the generator's performance.

