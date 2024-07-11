"""
Contains functions for training a PyTorch WGAN-GP model.
"""

import torch
import torch.nn as nn
import os

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from .model_utils import *
from .utils import *

def train_WGAN_GP(generator: nn.Module,
                  critic: nn.Module,
                  train_dataloader: torch.utils.data.DataLoader, 
                  val_dataloader: torch.utils.data.DataLoader, 
                  sample_indices: torch.Tensor,
                  gen_opt: torch.optim.Optimizer,
                  cr_opt: torch.optim.Optimizer,
                  epochs: int,
                  device: torch.device) -> Dict[str, List]:
    
    # loading hyperparameters
    params = load_hyperparameters(os.path.join("..", "config", "config.yaml"))

    Nr = params['nr']
    Nt = params['nt']
    L = params['l']
    T = params['t']
    NUM_ANTENNA_PAIRS = Nr * Nt

    BATCH_SIZE = params['batch_size']
    Z_DIM = params['z_dim']
    N_CRITIC = params['n_critic']
    LAMBDA_GP = params['lambda_gp']

    # setting up transmitted signal - unit power discrete impulse
    input_signal = torch.zeros(1, Nt, T, device=device, dtype=torch.complex128)
    input_signal[0,0,0] = 1
    input_signal[0,1,0] = 1 #12
    input_signal[0,2,0] = 1 #25
    input_signal[0,3,0] = 1 #39

    ij_matrix_full = torch.eye(NUM_ANTENNA_PAIRS, dtype=torch.float64, device=device).repeat(BATCH_SIZE, 1) 
    i_matrix_full = torch.eye(Nr, dtype=torch.float64, device=device).repeat(BATCH_SIZE, 1)

    generator.train()
    critic.train()

    c_losses = []
    g_losses = []

    for epoch in range(epochs):
        c_loss = g_loss = 0
        for batch_idx, batch_real in enumerate(tqdm(train_dataloader)):
            
            batch_real = batch_real.to(device) # shape [BATCH_SIZE, Nr, T+Nr]
            cur_batch_size = batch_real.shape[0]
            # setting up conditioning information
            ij_matrix = ij_matrix_full[:(cur_batch_size*NUM_ANTENNA_PAIRS)]
            i_matrix = i_matrix_full[:(cur_batch_size*Nr)] 

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(N_CRITIC):

                # generating a batch of fake data
                z = torch.randn(cur_batch_size*NUM_ANTENNA_PAIRS, Z_DIM, dtype=torch.float64, device=device)

                channel_tensor = generator(z, ij_matrix)
                batch_fake = get_fake_batch(input_signal, channel_tensor, sample_indices)
                
                # interleave real and imaginary
                batch_real_int = prepare_complex_signal(batch_real).view(cur_batch_size*Nr, -1)
                batch_fake_int = prepare_complex_signal(batch_fake).view(cur_batch_size*Nr, -1)

                # calculating critic loss
                critic_real = critic(batch_real_int, i_matrix).view(-1)
                critic_fake = critic(batch_fake_int, i_matrix).view(-1)
                gp = gradient_penalty(critic, batch_real, batch_fake, i_matrix)
                critic_loss = (-(torch.mean(critic_real) - torch.mean(critic_fake))) + (LAMBDA_GP * gp)
                c_loss += critic_loss.item()

                critic.zero_grad() # zero optimizer gradients
                critic_loss.backward(retain_graph=True) # back-prop
                cr_opt.step() # step the optimizer

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(batch_fake_int, i_matrix).view(-1)
            gen_loss = -(torch.mean(gen_fake))
            g_loss += gen_loss.item()

            generator.zero_grad() # zero optimizer gradients
            gen_loss.backward() # back-prop
            gen_opt.step()

        c_loss = c_loss / (N_CRITIC * len(train_dataloader))
        g_loss = g_loss / len(train_dataloader)
        c_losses.append(c_loss)
        g_losses.append(g_loss)

        print(f"Epoch [{epoch+1}/{epochs}] \ Loss D: {c_loss:.4f}, loss G: {g_loss:.4f}")

    return c_losses, g_losses