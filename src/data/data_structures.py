# Standard libraries 
import numpy as np

from typing import List, Tuple

# Deep learning imports
import torch

# My imports
from utils.utils import normalize, minmaxscale

##############################
# Class: create dataset for updating statistics
##############################
class Dataset:
  def __init__(self, dataU: np.ndarray=np.array([]), dataY: np.ndarray=np.array([]), dataG: np.ndarray=np.array([])) -> None:
    self.dataU = dataU
    self.dataY = dataY
    self.dataG = dataG

  def update_statistics(self) -> None:
    assert self.dataU.shape[0]

    # for Brach input
    self.dataU_mean = np.mean(self.dataU, axis=0)
    self.dataU_std = np.std(self.dataU, axis=0)
    self.dataU_min = np.min(self.dataU, axis=0)
    self.dataU_max = np.max(self.dataU, axis=0)

    # for DON output
    self.dataG_mean = np.mean(self.dataG, axis=0)
    self.dataG_std = np.std(self.dataG, axis=0)
    self.dataG_min = np.min(self.dataG, axis=0)
    self.dataG_max = np.max(self.dataG, axis=0)

##############################
# Function: get distributed dataset
##############################
def get_distributed_dataset(data: np.ndarray, num_clients: int, verbose: bool=True):
  num_items = int(len(data) / num_clients)
  dict_clients, all_idxs = {}, [i for i in range(len(data))]
  for i in range(num_clients):
    dict_clients[i] = set(np.arange(i * num_items, (i+1) * num_items))   
    all_idxs = list(set(all_idxs) - dict_clients[i])
  if verbose:
    print(f"The number of items per client is {num_items}")
  return dict_clients

##############################
# Function: get one triplet - FWI
##############################
def get_one_triplet(input: np.ndarray, output: np.ndarray, ts: np.ndarray, offsets: np.ndarray, i: int, j: int) -> Tuple[np.ndarray]:
  return input, np.array([ts[i],offsets[j]]), np.array([output[i,j]])

##############################
# Function: get FWI data from one input
##############################
def get_FWI_data_from_one_input(input: np.ndarray, output: np.ndarray, ts: np.ndarray, offsets: np.ndarray):
  u, y, G = [], [], []
  nt = output.shape[0]
  nx = output.shape[-1]
  for i in range(nt):
    for j in range(nx):
       u_i, y_i, G_i = get_one_triplet(input, output, ts, offsets, i, j)
       u.append(u_i)
       y.append(y_i)
       G.append(G_i)
       del u_i, y_i, G_i

  return np.stack(u), np.stack(y), np.stack(G)

##############################
# Function: get FWI client data
##############################
def get_FWI_client_data(input_data: np.ndarray, output_data: np.ndarray, ts: np.ndarray, offsets: np.ndarray) -> Tuple[np.ndarray]:
  # allocate memory for DON data
  u, y, G = [], [], []

  for sample in range(input_data.shape[0]):
    u_i, y_i, G_i = get_FWI_data_from_one_input(input_data[sample,:].flatten(), output_data[sample,...], ts, offsets)
    u.append(u_i)
    y.append(y_i)
    G.append(G_i)
    del u_i, y_i, G_i

  u = np.vstack(u)
  y = np.vstack(y)
  G = np.vstack(G)

  return u, y, G

##############################
# Class: Operator data split
##############################
class Operator_dataset_split(torch.utils.data.Dataset):
  def __init__(self, data: Tuple, idxs: List) -> None:
    self.u, self.y, self.G = data
    self.idxs = [int(i) for i in idxs]
        
  def __len__(self) -> int:
    return len(self.idxs)
    
  def __getitem__(self, item: int) -> List:
    return (self.u[self.idxs[item],...], self.y[self.idxs[item],...], self.G[self.idxs[item],...])

##############################
# Class: Operator data split version 2
##############################
class Operator_dataset_split_v2(torch.utils.data.Dataset):
  def __init__(self, data: Tuple, client_number: int) -> None:
    self.u, self.y, self.G = data
    self.client = client_number
        
  def __len__(self) -> int:
    return self.u[self.client].shape[0]
    
  def __getitem__(self, item: int) -> List:
    return (self.u[self.client][item,...], self.y[self.client][item,...], self.G[self.client][item,...])