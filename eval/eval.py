import torch
import torch.nn as nn
import numpy as np
import yaml

from scipy.io import loadmat
from typing import Tuple

def get_test_data(path: str, var_name: str) -> torch.Tensor:
    """
    Loads and transposes a matrix from a MATLAB file.

    Parameters:
    path (str): The file path to the MATLAB .mat file.
    var_name (str): The name of the variable in the .mat file to load.

    Returns:
    torch.Tensor: The transposed matrix as a PyTorch tensor.
    """
    
    H = loadmat(path)[var_name]
    H = np.transpose(H, axes=(0, 3, 2, 1))

    return H

def get_batch_generated(model: nn.Module, batch_size: int, Nr: int, Nt: int, L: int, z_dim: int, device: torch.device) -> torch.Tensor:
    """
    Generate a batch of channel tensors using a specified neural network model.
    
    Parameters:
        model (nn.Module): The neural network model that generates channel matrices. It must have a 'device' attribute and accept inputs.
        batch_size (int): The number of samples to generate in one batch.
        Nr (int): Number of receiving antennas.
        Nt (int): Number of transmitting antennas.
        L (int): Number of paths in the multipath channel.
        z_dim (int): Dimensionality of the random noise input to the model.
    
    Returns:
        Tensor: A tensor representing the batch of generated channel tensors with shape (batch_size, Nr, Nt, L).
    """

    NUM_ANTENNA_PAIRS = Nr * Nt
    
    with torch.no_grad():  # Disable gradient calculation
        z = torch.randn(batch_size*NUM_ANTENNA_PAIRS, z_dim, dtype=torch.float64, device=device)
        ij_matrix = torch.eye(NUM_ANTENNA_PAIRS, dtype=torch.float64, device=device).repeat(batch_size, 1) 
        H = model(z, ij_matrix).view(batch_size, Nr, Nt, L)

    return H

def get_path_delays(path: str, var_name: str, delay_spread: float) -> np.ndarray:
    """
    Load normalized delay data from a YAML file and convert them to delays in seconds.

    Parameters:
        path (str): The file path to the YAML file containing delay data.
        var_name (str): The key in the YAML file to retrieve normalized delays.
        delay_spread (float): The delay spread used to scale the normalized delays.

    Returns:
        np.ndarray: An array of delays sorted in ascending order, in seconds.
    """
    # Load the YAML file
    with open(path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Extract the list of normalized delays
    normalized_delays = data[var_name]
    delay_spread = 300e-9
    delays_sec = np.sort(np.array(normalized_delays) * delay_spread)

    return delays_sec

def get_corr(H: np.ndarray, side_flag: bool) -> np.ndarray:
    """
    Compute the correlation matrix for a multipath channel matrix, either on the
    transmit side or receive side based on the side_flag.

    Parameters:
    H (np.ndarray): A 4D numpy array representing the channel matrix with dimensions 
                    [BATCH_SIZE, Nr, Nt, L], where:
                    - BATCH_SIZE is the number of batches,
                    - Nr is the number of receive antennas,
                    - Nt is the number of transmit antennas,
                    - L is the number of paths.
    side_flag (bool): A boolean flag to determine the type of correlation:
                      - True for transmit-side (Tx) correlation,
                      - False for receive-side (Rx) correlation.

    Returns:
    np.ndarray: A 2D numpy array of shape [Nt, Nt] or [Nr, Nr], depending on side_flag,
                containing the rounded mean correlation computed across batches and paths.
    """
    
    H = np.transpose(H, axes=(0,3,1,2))
    H_mh = np.transpose(np.conjugate(H), axes=(0, 1, 3, 2)) #Hermitian

    if side_flag:
        H_m = np.matmul(H_mh, H) # Tx-side Correlation
    else:
        H_m = np.matmul(H, H_mh) # Rx-side Correlation

    H_m = np.abs(np.sum(H_m, 1))
    H_m = np.round(np.mean(H_m, 0),2)

    return H_m

def compute_total_power_dB(mimo_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the average and total power in dB for a batch of MIMO channel tensors.

    Parameters:
    mimo_tensor (torch.Tensor): A complex tensor representing MIMO channels with shape [batch_size, Nr, Nt, L],
                    - BATCH_SIZE is the number of batches,
                    - Nr is the number of receive antennas,
                    - Nt is the number of transmit antennas,
                    - L is the number of paths.

    Returns:
    tuple(torch.Tensor, torch.Tensor):
        - Average power in dB per link (shape [Nr, Nt]): Average dB power for each Tx-Rx link across the batch.
        - Average total power in dB (scalar): The average of the total power in dB summed across all Tx-Rx pairs and batches.
    """
    
    power_per_path = torch.abs(mimo_tensor)**2
    total_power_per_link = torch.sum(power_per_path, dim=3)
    power_dB_per_link = 10 * torch.log10(total_power_per_link)
    
    # Handle potential -inf values due to log10(0)
    power_dB_per_link = torch.where(torch.isinf(power_dB_per_link), torch.tensor(0.0, device=power_dB_per_link.device), power_dB_per_link)

    avg_power_dB_per_link = torch.mean(power_dB_per_link, dim=0)
    avg_total_power_dB = torch.mean(avg_power_dB_per_link)

    return avg_power_dB_per_link, avg_total_power_dB

def compute_average_delay_per_link(delay_vector: torch.Tensor, mimo_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the average delay of the PDP per Tx-Rx link across all batches, and then computes the overall average delay across all links.

    Parameters:
    delay_vector (torch.Tensor): A tensor of shape [L] containing the delays for each path in seconds.
    mimo_tensor (torch.Tensor): A complex tensor representing MIMO channels with shape [batch_size, Nr, Nt, L].

    Returns:
    Tuple[torch.Tensor, torch.Tensor]:
        - average_delay_per_link (torch.Tensor): The average delay for each Tx-Rx link, shape [Nr, Nt].
        - overall_average_delay (torch.Tensor): A scalar tensor representing the average delay across all Tx-Rx links.
    """
    
    delay_vector = delay_vector.view(1,1,1,-1)  # Shape [1, 1, 1, L]
    
    power_per_path = torch.abs(mimo_tensor)**2
    weighted_delays = delay_vector * power_per_path
    
    total_power_per_link = torch.sum(power_per_path, dim=3)
    total_weighted_delays = torch.sum(weighted_delays, dim=3)
    
    # Avoid division by zero by ensuring no zero power values
    total_power_per_link = torch.where(total_power_per_link == 0, torch.tensor(1.0, device=total_power_per_link.device), total_power_per_link)
    
    average_delay_per_link_per_batch = total_weighted_delays / total_power_per_link
    
    average_delay_per_link = torch.mean(average_delay_per_link_per_batch, dim=0)
    overall_average_delay = torch.mean(average_delay_per_link)

    return average_delay_per_link, overall_average_delay

def compute_rms_delay_per_link(delay_vector: torch.Tensor, mimo_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the RMS delay spread of the PDP per Tx-Rx link across all batches, and then computes the overall average RMS delay across all links.

    Parameters:
    delay_vector (torch.Tensor): A tensor of shape [L] containing the delays for each path in seconds.
    mimo_tensor (torch.Tensor): A complex tensor representing MIMO channels with shape [batch_size, Nr, Nt, L].

    Returns:
    Tuple[torch.Tensor, torch.Tensor]:
        - rms_delay_per_link (torch.Tensor): The RMS delay spread for each Tx-Rx link, shape [Nr, Nt].
        - overall_average_rms_delay (torch.Tensor): A scalar tensor representing the average RMS delay across all Tx-Rx links.
    """

    power_per_path = torch.abs(mimo_tensor)**2
    total_power_per_link = torch.sum(power_per_path, dim=3)

    delay_vector = delay_vector.view(1, 1, 1, -1)  # Shape [1, 1, 1, L]
    weighted_delays = delay_vector * power_per_path
    weighted_delays_squared = (delay_vector ** 2) * power_per_path

    total_weighted_delays = torch.sum(weighted_delays, dim=3)
    total_weighted_delays_squared = torch.sum(weighted_delays_squared, dim=3)

    average_delay_per_link_per_batch = total_weighted_delays / total_power_per_link
    second_moment_delay = total_weighted_delays_squared / total_power_per_link

    variance_delay = second_moment_delay - (average_delay_per_link_per_batch ** 2)
    rms_delay_per_link_per_batch = torch.sqrt(variance_delay)

    rms_delay_per_link = torch.mean(rms_delay_per_link_per_batch, dim=0)

    overall_average_rms_delay = torch.mean(rms_delay_per_link)

    return rms_delay_per_link, overall_average_rms_delay

