'''
This file contains the code to train different variations of Generative Adversarial Networks (GANs) on time series data.
It includes classes for the CGANTrainer (trainer for Conditional GANs), GAN, and specific classes for three GAN variants: 
RCGAN, TimeGAN, and RCWGAN.
'''

import functools

import torch
from torch import autograd

from lib.algos.base import BaseAlgo
from lib.arfnn import ResFNN
from lib.utils import sample_indices


class CGANTrainer(object):
    def __init__(
            self,
            G,  # Generator network
            D,  # Discriminator network
            G_optimizer,  # Optimizer for the generator
            D_optimizer,  # Optimizer for the discriminator
            p,  # Time delay parameter
            q,  # Future time steps parameter
            gan_algo,  # Type of GAN algorithm to use
            reg_param: float = 10.  # Regularization parameter
    ):
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        self.p = p
        self.q = q

        self.gan_algo = gan_algo
        self.reg_param = reg_param

    def G_trainstep(self, x_fake, x_real):
        # This function performs one training step for the Generator
        toggle_grad(self.G, True)  # Enable gradients for G
        self.G.train()  # Set G to training mode
        self.G_optimizer.zero_grad()  # Zero out gradients in G
        d_fake = self.D(x_fake)  # Pass fake data through D
        self.D.train()  # Set D to training mode
        gloss = self.compute_loss(d_fake, 1)  # Compute generator loss
        if self.gan_algo == 'TimeGAN':  # If TimeGAN, add reconstruction loss
            gloss = gloss + torch.mean((x_fake - x_real) ** 2)
        gloss.backward()  # Backward pass
        self.G_optimizer.step()  # Update G's weights
        return gloss.item()  # Return the generator loss

    def D_trainstep(self, x_fake, x_real):
        # This function performs one training step for the Discriminator
        toggle_grad(self.D, True)  # Enable gradients for D
        self.D.train()  # Set D to training mode
        self.D_optimizer.zero_grad()  # Zero out gradients in D

        # Real data training
        x_real.requires_grad_()  # Enable gradients for real data
        d_real = self.D(x_real)  # Pass real data through D
        dloss_real = self.compute_loss(d_real, 1)  # Compute loss on real data

        # Fake data training
        x_fake.requires_grad_()  # Enable gradients for fake data
        d_fake = self.D(x_fake)  # Pass fake data through D
        dloss_fake = self.compute_loss(d_fake, 0)  # Compute loss on fake data

        # Compute total discriminator loss
        dloss = dloss_fake + dloss_real
        dloss.backward()  # Backward pass

        # Compute gradient penalty for WGAN-GP
        if self.gan_algo == 'RCWGAN':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake)
            reg.backward()
        else:
            reg = torch.ones(1)

        # Update discriminator parameters
        self.D_optimizer.step()

        # Disable gradient for D
        toggle_grad(self.D, False)
        return dloss_real.item(), dloss_fake.item(), reg.item()
    def compute_loss(self, d_out, target):
        # Compute the loss based on the GAN variant
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        if self.gan_algo in ['RCGAN', 'TimeGAN']:
            return torch.nn.functional.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_algo == 'RCWGAN':
            return (2 * target - 1) * d_out.mean()

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        # Compute gradient penalty for WGAN-GP
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake  # Interpolate between real and fake samples
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.D(x_interp)  # Pass interpolated samples through D
        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()  # Compute gradient penalty
        return reg


def toggle_grad(model, requires_grad):
    # Enable or disable gradients for a model
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    # Compute the squared gradient
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class GAN(BaseAlgo):
    # Inherits from the BaseAlgo class, an abstract class that defines the necessary components of a machine learning algorithm
    # This class will serve as the base class for specific GAN algorithms that will be defined later (RCGAN, TimeGAN, RCWGAN)

    def __init__(self, base_config, gan_algo, x_real):
        # Initialize the GAN with the provided configuration, algorithm type, and real data
        super(GAN, self).__init__(base_config, x_real)
        self.D_steps_per_G_step = 2  # Number of discriminator updates per generator update
        self.D = ResFNN(self.dim * (self.p + self.q), 1, self.hidden_dims, True).to(self.device)
        # Define Adam optimizers for G and D with TTUR (Two Time-scale Update Rule)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0, 0.9))
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas=(0, 0.9))

        self.gan_algo = gan_algo
        self.trainer = CGANTrainer(  # Initialize the CGANTrainer which will handle the training of the GAN
            G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer,
            gan_algo=gan_algo, p=self.p, q=self.q,
        )

    def step(self):
        # Method for performing a training step for the GAN
        for i in range(self.D_steps_per_G_step):
            # Generate fake data
            z = torch.randn(self.batch_size, self.q, self.latent_dim).to(self.device)
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_real[indices, :self.p].clone().to(self.device)
            with torch.no_grad():
                x_fake = self.G(z, x_past.clone())  # Generate fake data using the generator
                x_fake = torch.cat([x_past, x_fake], dim=1)
            # Update discriminator
            D_loss_real, D_loss_fake, reg = self.trainer.D_trainstep(x_fake, self.x_real[indices].to(self.device))
            if i == 0:
                self.training_loss['D_loss_fake'].append(D_loss_fake)
                self.training_loss['D_loss_real'].append(D_loss_real)
                self.training_loss['RCWGAN_reg'].append(reg)
        # Update generator
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_real[indices, :self.p].clone().to(self.device)
        x_fake = self.G.sample(self.q, x_past)
        x_fake_past = torch.cat([x_past, x_fake], dim=1)
        G_loss = self.trainer.G_trainstep(x_fake_past, self.x_real[indices].clone().to(self.device))
        self.training_loss['D_loss'].append(D_loss_fake + D_loss_real)
        self.training_loss['G_loss'].append(G_loss)
        self.evaluate(x_fake)  # Evaluate the model


class RCGAN(GAN,):
    # Inherits from the GAN class, specific to the RCGAN variant

    def __init__(self, base_config, x_real):
        super(RCGAN, self).__init__(base_config, 'RCGAN', x_real)  # Initialize RCGAN with the provided configuration and real data


class TimeGAN(GAN):
    # Inherits from the GAN class, specific to the TimeGAN variant

    def __init__(self, base_config, x_real):
        # Initialize TimeGAN with the provided configuration and real data
        super(TimeGAN, self).__init__(base_config, 'TimeGAN', x_real)


class RCWGAN(GAN, ):
    # Inherits from the GAN class, specific to the RCWGAN variant

    def __init__(self, base_config, x_real):
        # Initialize RCWGAN with the provided configuration and real data
        super(RCWGAN, self).__init__(base_config, 'RCWGAN', x_real)
