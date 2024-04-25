# Standard imports
import numpy as np

from scipy import interpolate
from typing import Any, Dict, List, Tuple

# Deep learning imports 
import torch
import torch.nn as nn

# Plotting imports
import matplotlib.pyplot as plt

# My imports
from data.ode_data import ode_system
from utils.utils import compute_l2_error


##############################
# Function: test an input u for ODE
##############################
def test_u_ode(system: Any, u: np.ndarray, T: int, num_sensors: int, num: int=100) -> List:
  sensors = np.linspace(0, T, num=num_sensors)[:, None]
  sensor_vals = u(sensors)
  y_test = np.linspace(0, T, num=num)[:, None]
  u_test = np.tile(sensor_vals.T, (num, 1))
  G_test = system.eval_s_fn(u, y_test)
  return u_test, y_test, G_test

#################################
# function: centralized inference
#################################
def centralized_inference(
    model: nn.Module,
    system: ode_system,
    spaces: Dict,
    client_number: int,
    num_test: int,
    num_sensors: int,
    T: float,
    device: torch.device=torch.device("cpu"),
    verbose: bool=True,
    ) -> Tuple:
  if verbose:
    print(f'\n***** Testing centralized model using {num_test} trajectories for space {client_number} *****\n')

  ##############################
  # allocate memory for L2_error
  ##############################
  L2_errors = []

  ####################
  # generate test data
  ####################
  features = spaces[client_number].random(num_test)
  sensors = np.linspace(0.0, T, num=num_sensors)[:, None]
  u = spaces[client_number].eval_batch(features, sensors)
  u_test, y_test, G_test = [], [], []
  for i in range(num_test):
    u_i = interpolate.interp1d(
      np.ravel(sensors), 
      u[i,...], 
      kind="cubic", 
      copy=False, 
      assume_sorted=True
      )
    u_test_i, y_test_i, G_test_i = test_u_ode(system,u_i, T, num_sensors)
    u_test.append(u_test_i)
    y_test.append(y_test_i)
    G_test.append(G_test_i)
    del u_test_i, y_test_i, G_test_i

  # allocate memory for predictions
  G_pred = []

  ######
  # test
  ######
  for i in range(len(u_test)):
    U_test_i = torch.tensor(u_test[i], dtype=torch.float32, device=device)
    Y_test_i = torch.tensor(y_test[i], dtype=torch.float32, device=device)
    with torch.no_grad():
      G_pred_k = model((U_test_i, Y_test_i))
    G_pred.append(G_pred_k.cpu().detach().numpy())

    # compute metric
    l2_error = compute_l2_error(G_test[i].flatten(), G_pred[-1].flatten())
    L2_errors.append(l2_error)

  ##########################
  # compute error statistics
  ##########################
  mean_error = np.mean(L2_errors) * 100
  std_error = np.std(L2_errors) * 100
  if verbose:
    print("The L2-error statistics for centralized model and space {} are given below".format(client_number))
    print("----------------------------")
    print("L2-err mean   L2-err std")
    print("----------------------------")
    print("{:.3f}        {:.3f}".format(mean_error, std_error))

  return mean_error, std_error, L2_errors

##########################
# function: D2NO inference
##########################
def D2NO_inference(
    models: Dict,
    inference_dataset: Tuple,
    client_number: int=0,
    device: torch.device=torch.device("cpu"),
    verbose: bool=True,
    ) -> Tuple[Dict, Tuple]:
  
  ########################
  # extract inference data
  ########################
  u_infer_clients, y_infer_clients, G_infer_np_clients = inference_dataset
  u_infer = u_infer_clients[client_number]
  y_infer = y_infer_clients[client_number]
  G_infer_np = G_infer_np_clients[client_number]
  num_infer = u_infer.shape[0]

  if verbose:
    print(f'\n***** Testing model {client_number} using {num_infer} samples*****\n')
  
  ################
  # define metrics
  ################
  metrics = {}
  metrics["L2_error"] = []

  ####################
  # run inference loop
  ####################
  for i in range(num_infer):
    
    # collect inputs and move to device
    u_i = u_infer[i,...].to(device)
    y_i = y_infer[i,...].to(device)

    # predict without computing gradients
    with torch.no_grad():
      G_pred_i = models[client_number]((u_i, y_i))

    # compute metrics
    L2_error = compute_l2_error(G_infer_np[i,...].flatten(), G_pred_i.detach().cpu().numpy().flatten())
    metrics["L2_error"].append(L2_error)

  # compute error statistics
  mean_error = np.mean(metrics["L2_error"]) * 100
  std_error = np.std(metrics["L2_error"]) * 100
  if verbose:
    print("\nThe mean L2-error for client {} is = {:.3f} %".format(client_number, mean_error))
    print("The standard deviation L2-error for client {} is = {:.3f} %".format(client_number, std_error))
  
  return metrics, (mean_error, std_error)

##############################
# Function: local inference of models
##############################
def local_inference(
    models: Dict,
    system: Any,
    spaces: Dict,
    num_test: int,
    num_sensors: List,
    client_number: int,
    T: float,
    device: torch.device=torch.device("cpu"),
    idxs_plot: List=[0,1,2,3],
    verbose: bool=False
    ) -> List:
  if verbose:
    print(f'\n***** Testing model {client_number} using {num_test} trajectories*****\n')
  
  # define metrics
  metrics = {}
  metrics["L2_error"] = []

  # generate test data
  features = spaces[client_number].random(num_test)
  sensors = np.linspace(0, T, num=num_sensors[client_number])[:, None]
  u = spaces[client_number].eval_batch(features, sensors)
  u_test, y_test, G_test = [], [], []

  for i in range(num_test):
    u_i = interpolate.interp1d(
        np.ravel(sensors), u[i,...], kind="cubic", copy=False, assume_sorted=True
        )
    u_test_i, y_test_i, G_test_i = test_u_ode(system, u_i, T, num_sensors[client_number])
    u_test.append(u_test_i)
    y_test.append(y_test_i)
    G_test.append(G_test_i)
    del u_test_i, y_test_i, G_test_i 

  # move the data to torch and test
  G_pred = []

  for i in range(len(u_test)):
    # predict with local model
    u_test_torch_i = torch.tensor(u_test[i], dtype=torch.float32, device=device)
    y_test_torch_i = torch.tensor(y_test[i], dtype=torch.float32, device=device)

    with torch.no_grad():
      G_pred_i = models[client_number]((u_test_torch_i, y_test_torch_i))
    G_pred.append(G_pred_i.cpu().detach().numpy())

    # compute metrics
    L2_error = compute_l2_error(G_test[i].flatten(), G_pred[-1].flatten())
    metrics["L2_error"].append(L2_error)

    # plot if required
    if i in idxs_plot:
      plt.figure()
      plt.plot(y_test[i].flatten(), G_test[i].flatten(), "b", lw=2.5, label="True solution")
      plt.plot(y_test[i].flatten(), G_pred[-1].flatten(), "r-", lw=1.5, label=f"Predicted by the {client_number + 1}th client")
      plt.legend(prop={'size':14})
      plt.ylabel("$G(u)(y)$", size=14)
      plt.xlabel("$y$", size=14)
      plt.show()
      if verbose:
        print("The L2-error of the {}-th example for client {} is = {:.3f} %".format(i, client_number, L2_error * 100))

  # compute error statistics
  mean_error = np.mean(metrics["L2_error"]) * 100
  std_error = np.std(metrics["L2_error"]) * 100
  if verbose:
    print("\nThe mean L2-error for client {} is = {:.3f} %".format(client_number, mean_error))
    print("The standard deviation L2-error for client {} is = {:.3f} %".format(client_number, std_error))
  
  return mean_error, std_error


