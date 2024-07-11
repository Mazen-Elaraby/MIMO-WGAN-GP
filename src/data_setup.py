"""
Contains functionality for creating PyTorch DataLoaders for 
Custom Dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from typing import Tuple

class MatDataset(Dataset):
    """
    Dataset class for loading numeric data from MATLAB .mat files stored as PyTorch tensors.

    Attributes:
        mat_file (str): Path to the MATLAB .mat file.
        variable_name (str): Name of the variable in the .mat file to load.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves the sample at the given index as a PyTorch tensor.
    """

    def __init__(self, mat_file: str, variable_name: str) -> None:
        """
        Initializes the dataset by loading data from a specified .mat file.

        Parameters:
            mat_file (str): Path to the MATLAB .mat file.
            variable_name (str): The key in the .mat file that contains the dataset.
        """
        self.data = loadmat(mat_file)[variable_name]

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves the item at the specified index.

        Parameters:
            idx (int): The index of the element to retrieve.

        Returns:
            torch.Tensor: The data sample at the specified index, converted to a PyTorch tensor.
        """
        item = self.data[idx]
        item_tensor = torch.from_numpy(item)
        return item_tensor    
''' 
class MatDataset(Dataset):
    def __init__(self, mat_file: str, variable_name: str, device: torch.device = None) -> None:
        self.device = device 
        # Load the dataset and convert to tensors
        mat_data = loadmat(mat_file)[variable_name]
        self.data = [torch.tensor(item, dtype=torch.complex128, device=self.device) for item in mat_data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
'''
def create_dataloaders(train_dataset_path: str, test_dataset_path: str, val_dataset_path: str, 
                       train_var_name: str, test_var_name: str, val_var_name: str,
                       batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates dataloaders for training, testing, and validation datasets.

    Parameters:
        train_dataset_path (str): Path to the training dataset .mat file.
        test_dataset_path (str): Path to the testing dataset .mat file.
        val_dataset_path (str): Path to the validation dataset .mat file.
        batch_size (int): The number of samples per batch to load.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for the training, testing, and validation datasets.
    """
    # Create datasets
    train_dataset = MatDataset(train_dataset_path, train_var_name)
    test_dataset = MatDataset(test_dataset_path, test_var_name)
    val_dataset = MatDataset(val_dataset_path, val_var_name)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, test_dataloader, val_dataloader