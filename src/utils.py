import torch
import yaml
import csv

from pathlib import Path
from typing import Dict, Any, List

def load_hyperparameters(filename: str) -> Dict[str, Any]:
    """
    Loads hyperparameters from a YAML file and returns them as a dictionary.

    Args:
        filename (str): The path to the YAML file containing the hyperparameters.

    Returns:
        Dict[str, Any]: A dictionary containing the hyperparameters loaded from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(filename, 'r') as file:
        try:
            params = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")
            raise
    return params


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  

def write_to_csv(c_losses: List[float], g_losses: List[float], filename: str) -> None:
    """
    Writes the discriminator and generator losses to a CSV file.

    Args:
    c_losses (List[float]): A list of discriminator losses recorded during training.
    g_losses (List[float]): A list of generator losses recorded during training.
    filename (str): The filename for the CSV output. Defaults to 'losses.csv'.

    Returns:
    None
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Critic Loss', 'Generator Loss'])  # Header row
        for c_loss, g_loss in zip(c_losses, g_losses):
            writer.writerow([c_loss, g_loss])

    print(f"[INFO] losses written to: {filename}")
