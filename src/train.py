"""
Trains a PyTorch WGAN_GP model using device-agnostic code.
"""

import torch
import os

from .data_setup import *
from .model_builder import *
from .utils import *
from .engine import *

def main():

    # loading hyperparameters
    params = load_hyperparameters(os.path.join("..", "config", "config.yaml"))

    NUM_EPOCHS = params['num_epochs']
    BATCH_SIZE = params['batch_size']
    NUM_WORKERS = params['num_workers']
    HIDDEN_UNITS = params['hidden_units']
    LEARNING_RATE = params['learning_rate']
    Z_DIM = params['z_dim']
    EMBED_DIM = params['embed_dim']

    # MIMO setting 
    Nr = params['nr'] # Number of receive antennas
    Nt = params['nt'] # Number of transmit antennas
    L = params['l'] # Number of paths in the multipath channel
    T = params['t'] # Number of samples in the ip/op signal

    # Setup directories
    train_dataset_path = os.path.join("..", "Dataset", "train_data_TDL_A.mat")
    test_dataset_path = os.path.join("..", "Dataset", "test_data_TDL_A.mat")
    val_dataset_path = os.path.join("..", "Dataset", "val_data_TDL_A.mat")

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, _, val_dataloader = create_dataloaders(train_dataset_path, test_dataset_path, val_dataset_path,
                                                                        "rx_train_data", "rx_test_data", "rx_val_data",
                                                                            BATCH_SIZE, NUM_WORKERS)

    # Intialize Models (Generator & Critic)
    generator = Generator(Nr=Nr, Nt=Nt, l=L, z_dim=Z_DIM, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_UNITS).to(device)
    generator = generator.double()  # Converts all parameters to torch.float64

    critic = Critic(N=T+Nr, num_receive_antennas=Nr, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_UNITS).to(device)
    critic = critic.double() # Converts all parameters to torch.float64

    # Setup optimizers
    gen_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    cr_opt = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # train models
    sample_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 17, 19, 20, 22, 23, 28, 35, 38, 42, 44, 46, 49, 89], device=device) 
    c_losses, g_losses = train_WGAN_GP(generator, critic, train_dataloader, val_dataloader, sample_indices, gen_opt, cr_opt, NUM_EPOCHS, device)

    # write losses to csv file
    write_to_csv(c_losses, g_losses, os.path.join("..", "Logs", "losses.csv"))

    # save models
    save_model(model=generator, target_dir="../models", model_name="G.pt")

    save_model(model=critic, target_dir="../models", model_name="C.pt")
    

if __name__ == '__main__':
    main()