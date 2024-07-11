"""
Contains PyTorch model code to instantiate a WGAN-GP model.
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, Nr: int, Nt: int, l: int, z_dim: int, embed_dim: int = 4, hidden_dim: int = 100):
        """
        Generator network for MIMO channel modeling using WGAN-GP.

        Args:
        Nr (int): Number of receive antennas.
        Nt (int): Number of transmit antennas.
        l (int): Number of paths in the multipath channel.
        z_dim (int): Dimension of the latent space vector.
        embed_dim (int): Dimensionality of the embedding space for each antenna pair.
        hidden_dim (int): Dimensionality of the hidden layers in the MLP.
        """
        super(Generator, self).__init__()

        self.Nr = Nr
        self.Nt = Nt
        self.num_antenna_pairs = self.Nr * self.Nt
        self.l = l

        self.embedding = nn.Linear(self.num_antenna_pairs, embed_dim)
        self.main = nn.Sequential(
            nn.Linear(z_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*self.l) #l-elements for the real and the imaginary (alternating)
        )

    def forward(self, z: torch.Tensor, ij: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
        z (torch.Tensor): Batch of latent vectors.
        ij (torch.Tensor): Batch of joint one-hot encoded vectors for antenna pairs.

        Returns:
        torch.Tensor: Batch of generated channel vectors.
        """
        ij_embedded = self.embedding(ij)
        combined_input = torch.cat((z, ij_embedded), dim=1)
        output = self.main(combined_input)

        complex_output = output.view(-1, self.l, 2)
        complex_output = torch.view_as_complex(complex_output)
        complex_output = complex_output.view(-1, self.Nr, self.Nt, self.l)

        return complex_output


class Critic(nn.Module):
    def __init__(self, N: int, num_receive_antennas: int, embed_dim: int = 4, hidden_dim: int = 100):
        """
        Discriminator network for MIMO channel modeling using WGAN-GP.

        Args:
        N (int): Length of the input signal vector + grammian vector.
        num_receive_antennas (int): Total number of receive antennas.
        embed_dim (int): Dimensionality of the embedding space for the receive antenna index.
        hidden_dim (int): Dimensionality of the hidden layers in the MLP.
        """
        super(Critic, self).__init__()
        self.embedding = nn.Linear(num_receive_antennas, embed_dim)
        self.main = nn.Sequential(
            nn.Linear((2*N) + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, signal: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
        signal (torch.Tensor): Batch of received signal vectors.
        i (torch.Tensor): Batch of one-hot encoded vectors for receive antenna indices.

        Returns:
        torch.Tensor: Batch of outputs representing the authenticity of each signal.
        """
        i_embedded = self.embedding(i)
        combined_input = torch.cat((signal, i_embedded), dim=1)
        authenticity = self.main(combined_input)
        return authenticity
