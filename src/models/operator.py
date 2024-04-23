# Standard imports

from typing import Dict, List, Tuple

# Deep learning imports
import torch
import torch.nn as nn

# My imports
from data.data_structures import Operator_dataset_split, Operator_dataset_split_v2
from models.nns import FF_FNN, FNN, modified_FNN, old_FNN, old_modified_FNN
from utils.utils import MSE


##############################
# Class: the old classical DON
##############################
class old_classical_DON(nn.Module):
  def __init__(self, branch: Dict, trunk: Dict, use_bias: bool=True) -> None:
    super(old_classical_DON, self).__init__()
    # branch
    if branch["type"] == "FNN":
      self.branch = old_FNN(branch["layer_size"], branch["act_fn_name"])
    elif branch["type"] == "modified":
      self.branch = old_modified_FNN(branch["layer_size"], branch["act_fn_name"])
    elif branch["type"] == "Fourier":
      self.branch = FF_FNN(branch["layer_size"], branch["act_fn_name"], branch["freqs"])

    # trunk
    if trunk["type"] == "FNN":
      self.trunk = old_FNN(trunk["layer_size"], trunk["act_fn_name"])
    elif trunk["type"] == "modified":
      self.trunk = old_modified_FNN(trunk["layer_size"], trunk["act_fn_name"])

    self.use_bias  = use_bias
    if use_bias:
      self.tau = nn.Parameter(torch.rand(1), requires_grad=True)
    

  def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
    u, y = x
    B = self.branch(u)
    T = self.trunk(y)
    G = torch.einsum("bi, bi -> b", B, T)
    G = torch.unsqueeze(G, dim=-1)
    return G + self.tau if self.use_bias else G


##############################
# Class: the classical DON
##############################
class classical_DON(nn.Module):
  """Classical DeepONet implementation."""
  def __init__(self, branch_params: Dict, trunk_params: Dict, use_bias: bool=True) -> None:
    super(classical_DON, self).__init__()

    if branch_params["type"] == "FNN":
      self.branch = FNN(branch_params) 
    elif branch_params["type"] == "modified": 
      self.branch = modified_FNN(branch_params)
    elif branch_params["type"] == "Fourier":
      self.branch = FF_FNN(branch_params)

    self.trunk = FNN(trunk_params) if trunk_params["type"] == "FNN" else modified_FNN(trunk_params)

    self.use_bias  = use_bias
    if use_bias:
      self.tau = nn.Parameter(torch.rand(1), requires_grad=True)


  def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
    u, y = x
    B = self.branch(u)
    T = self.trunk(y)
    G = torch.einsum("bi, bi -> b", B, T)
    G = torch.unsqueeze(G, dim=-1)
    return G + self.tau if self.use_bias else G


##############################
# Class: DON local update
##############################
class DON_local_update(object):
  def __init__(
    self, 
    params: Dict, 
    dataset: Tuple, 
    idxs: Dict, 
    logger: Dict, 
    client_number: int=0,
    device: torch.device=torch.device("cpu"), 
    ) -> None:
    
    self.params = params
    self.device = device
    self.trainloader = self.get_train_local(dataset, list(idxs), client_number)

  def get_train_local(self, dataset: Tuple, idxs: List, client_number: int=0) -> torch.utils.data.DataLoader:
    if not self.params["different_num_sensors"]:
      data = Operator_dataset_split(dataset, idxs)
    else:
      data = Operator_dataset_split_v2(dataset, client_number)
    
    trainloader = torch.utils.data.DataLoader(data, batch_size=self.params["batch_size"], shuffle=True) 
    return trainloader

  def update_weights(self, model: torch.nn, global_round: int, sharing_mode: str="all") -> List:
    # set mode to train local DON
    model.train()
    logger = {}
    logger["train_loss"] = []

    # set optimizer for local updates
    if self.params["optimizer"] == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), lr=self.params["lr"], momentum=0.5)
    elif self.params["optimizer"] == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=self.params["lr"])
    
    # local training loop
    for epoch in range(self.params["local_epochs"]):
      local_epoch_loss = 0
      for batch_idx, (u_batch, y_batch, G_batch) in enumerate(self.trainloader):
        model.zero_grad()
        u_batch = u_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        G_batch = G_batch.to(self.device)
        G_pred = model((u_batch, y_batch))
        loss = MSE(G_batch, G_pred)
        loss.backward()
        optimizer.step()
        local_epoch_loss += loss.detach().cpu().numpy().squeeze()

        # log the number of total gedients taken
        logger["total_gradients"] += sum(p.grad.numel() for p in model.parameters() if p.grad is not None)

        if self.params["verbose_local"] and (batch_idx % 50 == 0):
          print('| Global Round : {} | Local Epoch : {} | Loss: {:.3e}'.format(global_round, epoch, loss.item()))

      try:
        avg_epoch_loss = local_epoch_loss / len(self.trainloader)
      except ZeroDivisionError as e:
        print("error: ", e, "batch size larger than number of training examples")

      logger["train_loss"].append(avg_epoch_loss)
    
    if sharing_mode == "all":
      return model.state_dict(), sum(logger["train_loss"]) / len(logger["train_loss"])
    else:
      return model.branch.state_dict(), model.trunk.state_dict(), sum(logger["train_loss"]) / len(logger["train_loss"])


