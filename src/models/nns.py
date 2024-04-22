# Standard imports

from typing import Any, Dict, List

# Deep learning imports
import torch
import torch.nn as nn


##############################
# Class: Base activation function class
##############################
class ActivationFunction(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.name = self.__class__.__name__
    self.config = {"name": self.name}

##############################
# Class: Sin activation function
##############################
class Sine(ActivationFunction):
  """Sine activation function."""
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)

##############################
# Activation function by name dir
##############################
act_fn_by_name = {
  "tanh": nn.Tanh(),
  "relu": nn.ReLU(),
  "leaky": nn.LeakyReLU(),
  "gelu": nn.GELU(),
  "sigmoid": nn.Sigmoid(),
  "sin": Sine(),
  }

##############################
# Class: Fourier network
##############################
class FF_FNN(nn.Module):
  """Fourier feature feedforward neural network"""
  def __init__(self, params: Dict) -> None:
    super(FF_FNN, self).__init__()
    layer_size = params["layer_size"]
    act_fn_name = params["act_fn_name"]
    act_fn = act_fn_by_name[act_fn_name]
    freqs = params["freqs"]
    self.FF = freqs * torch.randn(layer_size[0], layer_size[1] // 2)
    self.net = nn.ModuleList()
    for k in range(1, len(layer_size) - 2):
      self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
      self.net.append(act_fn)
    self.net.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))
    self.net.apply(self._init_weights)

  def _init_weights(self, m: Any) -> None:
    if isinstance(m, nn.Linear):
      nn.init.xavier_normal_(m.weight)
      m.bias.data.zero_()
  
  def input_encoding(self, x, w):
    out = torch.hstack((torch.sin(torch.mm(x,w)),torch.cos(torch.mm(x,w))))
    return out

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = self.input_encoding(x, self.FF.to(x.device))
    for k in range(len(self.net)):
      y = self.net[k](y)
    return y

##############################
# Class: Feedforward neural network
##############################
class FNN(nn.Module):
  """Feedforward neural network."""
  def __init__(self, params: Dict, use_act_last: bool=False) -> None:
    super(FNN, self).__init__()
    layer_size = params["layer_size"]
    act_fn_name = params["act_fn_name"]
    act_fn = act_fn_by_name[act_fn_name]
    blocks = []
    for k in range(len(layer_size) - 2):
      blocks.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
      blocks.append(act_fn)

    blocks.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))
    if use_act_last:
      blocks.append(act_fn)
    self.net = nn.Sequential(*blocks)
    self.net.apply(self._init_weights)

  def _init_weights(self, m: Any) -> None:
    if isinstance(m, nn.Linear):
      nn.init.xavier_normal_(m.weight)
      m.bias.data.zero_()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

##############################
# Class: Old Feedforward neural network
##############################
class old_FNN(nn.Module):
  def __init__(self, layer_size: List, act_fn_name: str) -> None:
    super(old_FNN, self).__init__()
    act_fn = act_fn_by_name[act_fn_name]
    self.net = nn.ModuleList()
    for k in range(len(layer_size) - 2):
      self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
      self.net.append(act_fn)
    self.net.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))
    self.net.apply(self._init_weights)

  def _init_weights(self, m: Any) ->  None:
    if isinstance(m, nn.Linear):
      nn.init.xavier_normal_(m.weight)
      m.bias.data.zero_()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = x
    for k in range(len(self.net)):
      y = self.net[k](y)
    return y

##############################
# Class: old modified Feedforward neural network
##############################
class old_modified_FNN(nn.Module):
  def __init__(self, layer_size: List, act_fn_name: str) -> None:
    super(old_modified_FNN, self).__init__()
    self.net = nn.ModuleList()
    self.U = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
    self.V = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
    self.act_fn = act_fn_by_name[act_fn_name]
    for k in range(len(layer_size) - 1):
      self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
    self.net.apply(self._init_weights)
    self.U.apply(self._init_weights)
    self.V.apply(self._init_weights)

  def _init_weights(self, m: Any) ->  None:
    if isinstance(m, nn.Linear):
      nn.init.xavier_normal_(m.weight)
      m.bias.data.zero_()

  def forward(self, x):
    u = self.act_fn(self.U(x))
    v = self.act_fn(self.V(x))
    for k in range(len(self.net) - 1):
      y = self.net[k](x)
      y = self.act_fn(y)
      x = y * u + (1 - y) * v
    y = self.net[-1](x)
    return y

##############################
# Class: Modified Feedforward neural network
##############################
class modified_FNN(nn.Module):
  """Modified feedforward neural network."""
  def __init__(self, params: Dict) -> None:
    super(modified_FNN, self).__init__()
    layer_size = params["layer_size"]
    act_fn_name = params["act_fn_name"]
    self.net = nn.ModuleList()
    self.act_fn = act_fn_by_name[act_fn_name]
    self.U = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
    self.V = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
    
    for k in range(len(layer_size) - 1):
      self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))

    self.net.apply(self._init_weights)
    self.U.apply(self._init_weights)
    self.V.apply(self._init_weights)
            
  def _init_weights(self, m: Any) -> None:
    if isinstance(m, nn.Linear):
      nn.init.xavier_normal_(m.weight)
      m.bias.data.zero_()   

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    u = self.act_fn(self.U(x))
    v = self.act_fn(self.V(x))
    for k in range(len(self.net) - 1):
      y = self.net[k](x)
      y = self.act_fn(y)
      x = y * u + (1 - y) * v
    y = self.net[-1](x)
    return y