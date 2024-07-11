"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
import torch.nn as nn
from torch.nn.functional import conv1d

def prepare_tensor_for_conv(H: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
    """
    Prepares a destination tensor for convolution by broadcasting and copying 
    values from the input tensor H based on provided sample indices.

    Args:
        H (Tensor): The input tensor with shape (batch_size, Nr, Nt, l), where:
            - batch_size is the number of batches.
            - Nr is the number of receive antennas.
            - Nt is the number of transmit antennas.
            - l is the length of the channel paths.
        sample_indices (Tensor): A 1D tensor containing the indices of the samples.

    Returns:
        Tensor: The prepared destination tensor with shape 
        (batch_size, Nr, Nt, channel_vector_length), where channel_vector_length
        is determined by the maximum value in sample_indices plus one.
    """
    
    # Find the required length for the final tensor
    channel_vector_length = max(sample_indices).item() + 2

    # Extracting Dimensions
    batch_size, Nr, Nt, _ = H.shape
    device = H.device

    # Initialize the destination tensor with zeros
    destination_tensor = torch.zeros((batch_size, Nr, Nt, channel_vector_length), dtype=torch.complex128, device=device)

    # Create index tensors
    batch_indices = torch.arange(batch_size, device=device).view(-1,1,1,1)
    Nr_indices = torch.arange(Nr, device=device).view(1, -1, 1, 1)  
    Nt_indices = torch.arange(Nt, device=device).view(1, 1, -1, 1)  
    sample_indices = sample_indices.view(1, 1, 1, -1)  

    # Use broadcasting to copy the original tensor's paths into the destination tensor
    destination_tensor[batch_indices, Nr_indices, Nt_indices, sample_indices] = H

    return destination_tensor.flip(dims=[-1])


def mimo_conv_batched(input_signal: torch.Tensor, H: torch.Tensor)->torch.Tensor:
    """
    Perform a batched MIMO (Multiple Input Multiple Output) convolution on the input signal using the provided channel tensor.

    This function efficiently computes the convolution across a batch of samples using a broadcast operation, 
    applying the convolutional operation over multiple receive antennas simultaneously.

    Args:
        input_signal (torch.Tensor): A 3D input signal tensor of shape (1, Nt, T) where:
            - Nt: Number of transmit antennas
            - T: Length of the input signal for each antenna

        H (torch.Tensor): A 4D tensor representing the channel model of shape (batch_size, Nr, Nt, l) where:
            - batch_size: Number of samples in the batch
            - Nr: Number of receive antennas
            - Nt: Number of transmit antennas
            - l: Number of paths in the multipath channel

    Returns:
        torch.Tensor: An output signal matrix of shape (batch_size, Nr, T) containing the convolved received signals.
    """
    
    # Extracting dimensions
    batch_size, Nr, Nt, l = H.shape
    T = input_signal.shape[2]
    
    H = H.view(batch_size * Nr, Nt, l)  # Reshape H for conv1d: [(batch_size * Nr), Nt, l]
    Y = conv1d(input_signal, H, padding='same')
    Y = Y.view(batch_size, Nr, T)

    return Y


def prepare_complex_signal(complex_signal: torch.Tensor) -> torch.Tensor:
    """
    Prepares a complex signal for input into the neural network by interleaving
    the real and imaginary parts.

    Args:
        complex_signal (torch.Tensor): A complex-valued tensor of shape (batch_size, N_r, T)
                                       where N_r is the number of receive antennas and T is the time dimension.

    Returns:
        torch.Tensor: A real-valued tensor of shape (batch_size, N_r, 2*T) with interleaved real and imaginary parts.
    """
    # Separate real and imaginary parts
    real_part = complex_signal.real
    imag_part = complex_signal.imag

    # Interleave real and imaginary parts
    # Create an empty tensor of the appropriate shape to hold the interleaved data
    interleaved_signal = torch.zeros(real_part.size(0), real_part.size(1), 2 * real_part.size(2), dtype=real_part.dtype, device=real_part.device)

    # Place real and imaginary parts in the correct slots
    interleaved_signal[:, :, 0::2] = real_part  # Real parts are placed at even indices
    interleaved_signal[:, :, 1::2] = imag_part  # Imaginary parts are placed at odd indices

    return interleaved_signal


def gradient_penalty(critic: nn.Module, real_signals: torch.Tensor, fake_signals: torch.Tensor, i_indices: torch.Tensor) -> float:
    """
    Calculate the gradient penalty for the critic in a WGAN-GP.

    Args:
    critic (nn.Module): The critic network.
    real_signals (torch.Tensor): Real signal samples, shaped (batch_size, Nr, N).
    fake_signals (torch.Tensor): Generated signal samples, shaped (batch_size, Nr, N).
    i_indices (torch.Tensor): Batch of one-hot encoded vectors for receive antenna indices.

    Returns:
    float: Gradient penalty.
    """
    BATCH_SIZE, Nr, N = real_signals.shape
    device = real_signals.device

    epsilon = torch.rand(BATCH_SIZE, 1, 1, device=device).expand_as(real_signals)
    interpolated_signals = (epsilon * real_signals) + ((1 - epsilon) * fake_signals)

    # Calculate critic scores
    interpolated_signals = prepare_complex_signal(interpolated_signals).view(BATCH_SIZE*Nr, -1)
    interpolated_scores = critic(interpolated_signals, i_indices)

    # Take the gradient of the scores with respect to the images
    gradients = torch.autograd.grad(
        inputs=interpolated_signals,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores, device=device),
        create_graph=True,
        retain_graph=True
    )[0]

    # Flatten the gradients to calculate norm
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_norm = torch.norm(gradients, p=2, dim=1)

    # Calculate gradient penalty as mean of squares of (norms - 1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def get_fake_batch(input_signal: torch.Tensor, channel_tensor: torch.Tensor, sample_indices: torch.Tensor):
    """
    Generate a batch of fake data for training a WGAN-GP model.

    This function applies a convolutional operation to the input signals using the provided channel tensors, 
    modified based on sample indices. It then computes the Hermitian transpose of the output signals to form a Grammian matrix. 
    The final output is a concatenation of the output signals and their corresponding Grammian matrices along the channel dimension.

    Parameters:
    - input_signal (torch.Tensor): A tensor representing the input signals. 
    - channel_tensor (torch.Tensor): A tensor representing the channel effects. Should be of shape [batch_size, Nr, Nt, L], where:
        - batch_size: the number of samples in the batch.
        - Nr: Number of receiving channels.
        - Nt: Number of transmitting channels.
        - L: Number of paths in the multipath channel.
    - sample_indices (torch.Tensor): Indices used for modifying the channel tensors in preparation for convolution.

    Returns:
    - batch_fake (torch.Tensor): The fake data batch, which includes both the output signals and the Grammian matrices. 
      Shape [batch_size, Nr, T + Nr], where T is the length of output signals post convolution.
    """

    channel_tensor = prepare_tensor_for_conv(channel_tensor, sample_indices)
    
    output_signal = mimo_conv_batched(input_signal, channel_tensor) # shape [BATCH_SIZE, Nr, T]
    
    output_signal_herm = output_signal.mH # hermitian of the output signal shape [BATCH_SIZE, T, Nr]
    grammian = torch.matmul(output_signal, output_signal_herm) # grammian matrix shape [BATCH_SIZE, Nr, Nr]

    batch_fake = torch.cat((output_signal, grammian), dim=2) # shape [BATCH_SIZE, Nr, T+Nr]
    
    return batch_fake