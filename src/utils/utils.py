# Standard imports 
import numpy as np
import random 

from scipy import interpolate
from typing import Dict, Tuple

# Deep learning imports
import torch

# Plotting imports
import matplotlib.pyplot as plt 

##############################
# Function: normalize the data
##############################
def normalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float=1e-8) -> np.ndarray:
  return (data-mean)/(std+eps)

##############################
# Function: minmaxscale the data
##############################
def minmaxscale(data, min_val, max_val):
  return (data - min_val) / (max_val - min_val)

##############################
# unnormalize data
##############################
def unnormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return data*std+mean

##############################
# Function: compute trajectory error
##############################
def compute_l2_error(true_traj: np.ndarray, pred_traj: np.ndarray) -> float:
  L2_error = L2_rel_error(true_traj.flatten(), pred_traj.flatten())
  return L2_error

##############################
# Function: plot input function examples
##############################
def distributed_plot_input(
  spaces: Dict,
  T: int,
  num_sensors: Tuple,
  num_clients: int) -> None:

  # define sensors
  fine_grid = np.linspace(0, T, num=200)[:,None]

  for i in range(num_clients):
    sensors = np.linspace(0, T, num=num_sensors[i])[:, None]
    features = spaces[i].random(1)
    u = spaces[i].eval_batch(features, sensors)
    u_i = interpolate.interp1d(
      np.ravel(sensors),
      u,
      kind="cubic",
      copy=False,
      assume_sorted=True,
    )
    sensor_vals = u_i(sensors)
    u_vals = u_i(fine_grid)
    plt.plot(fine_grid.flatten(), u_vals.flatten(), lw=2.5, label=f"Input function sampled from space {i + 1}")
    plt.plot(sensors.flatten(), sensor_vals.flatten(), 's', label=f"distributed sampling - {num_sensors[i]} sensors")
    plt.legend(prop={'size':14})
    plt.ylabel("$u(t)$", size=14)
    plt.xlabel("$t$ (s)", size=14)

  plt.show()

##############################
# Function: get number of parameters
##############################
def get_num_params(model: torch.nn.Module) -> int:
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

##############################
# Function: trajectory L2_error
##############################
def L2_rel_error(true_traj: np.ndarray, pred_traj: np.ndarray) -> float:
  return np.linalg.norm(true_traj - pred_traj) / np.linalg.norm(true_traj)

##############################
# Function: MSE loss
##############################
def MSE(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
  return torch.mean((y_true - y_pred) ** 2)

##############################
# Function: Set seed
##############################
def set_seed(seed: int) -> None:
	"""
	Set seed for RNG
	Args:
	----------------
	  seed: int for init the RNG
	"""
	np.random.seed(seed)
	random.seed(seed)	
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	# Ensure that all operations are deterministic on GPU (if used) for reproducibility
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

#########################
# Function: restore torch model
#########################
def restore(restore_path):
  return torch.load(restore_path)


#########################
# Function: save torch model
#########################
def save(model: torch.nn, save_path: str) -> None:
  state = {
    'state_dict': model.state_dict(),
    }
  torch.save(state, save_path)