# Standard imports
import copy
import numpy as np

from tqdm.auto import trange
from typing import Any, Dict, List, Tuple

# Deep learning imports
import torch
import torch.nn as nn

# My imports 
from data.data_structures import OperatorDataset
from models.operator import DON_local_update
from utils.utils import compute_l2_error, MSE, unnormalize, save


####################################
# function: centralized DON training
####################################
def centralized_train(
  model: nn.Module,
  dataset: OperatorDataset,
  params: Dict,
  device: torch.device=None,
  verbose: bool=True
  ) -> Dict:

  if verbose:
    print(f'\n***** Training with Adam Optimizer for {params["num_epochs"]} epochs and using {dataset.len} data samples*****\n')

  ######################
  # define the optimizer
  ######################
  optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

  ##################
  # load the dataset
  ##################
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

  ########################
  # define logger and pbar
  ########################
  logger = {}
  logger["loss"] = []
  logger["best_loss"] = np.Inf
  logger["total_gradients"] = 0
  pbar = trange(params["num_epochs"])

  ###############
  # training loop 
  ###############
  for epoch in pbar:
    epoch_loss = 0
    model.train()

    # batch training
    for x_batch, y_batch in trainloader:
      # move data to device
      u, y = x_batch 
      x_batch = (u.to(device), y.to(device))
      y_batch = y_batch.to(device)

      # forward pass
      y_pred = model(x_batch)

      # compute loss
      loss = MSE(y_batch, y_pred)

      # compute gradient and backpropagate
      optimizer.zero_grad()
      loss.backward()

      # optimize
      epoch_loss += loss.detach().cpu().numpy().squeeze()
      optimizer.step()

      # log the number of gradients taken
      logger["total_gradients"] += sum(p.grad.numel() for p in model.parameters() if p.grad is not None)

    try: 
      avg_epoch_loss = epoch_loss / len(trainloader)
    except:
      print("error: batch size larger than number of training examples")

    # log epoch loss
    logger["loss"].append(avg_epoch_loss)

    # update best loss
    if avg_epoch_loss < logger["best_loss"]:
      logger["best_loss"] = avg_epoch_loss
      save(model, save_path="./output/centralized-best-model.pt")

    # print
    pbar.set_postfix({'Train Loss': np.round(avg_epoch_loss,6), 'Best Train Loss': np.round(logger["best_loss"],6), 'Gradients': logger["total_gradients"]})

  print(f"\n******** training summary ********\n")
  print(f'model best loss ={np.round(logger["best_loss"],6)}')
  print(f'\nthe number of gradients computed is={logger["total_gradients"]}')

  pbar.close()
  return logger



#########################
# Function: Average weights
#########################
def average_weights(w: List) -> Any:
  w_avg = copy.deepcopy(w[0])
  for key in w_avg.keys():
    for i in range(1, len(w)):
      w_avg[key] += w[i][key]
    w_avg[key] = torch.div(w_avg[key], len(w))
  return w_avg

##############################
# Function: D2NO training
##############################
def D2NO_train(
  models: Dict,
  dataset: Tuple,
  num_clients: int,
  params: Dict,
  client_groups: Dict,
  device: torch.device = torch.device("cpu"),
  save_name: str="./output/best-model-",
  ) -> Dict:
  ##############################
  # Step 1: unpack the dataset
  ##############################
  u_train, y_train, G_train = dataset 

  ##############################
  # Step 2: copy weights
  ##############################
  global_weights = models[0].state_dict()

  ##############################
  # Step 3: define logger
  ##############################
  logger = {}
  local= {}
  logger["total_gradients"] = 0
  logger["train_loss"] = []
  
  logger["best_loss"] = {} 
  for i in range(num_clients):
    logger["best_loss"][i] = np.Inf
  logger["best_global_loss"] = np.Inf

  logger["L2-error"] = {}
  logger["best-L2-error"] = {}
  for i in range(num_clients):
    logger["L2-error"][i] = []
    logger["best-L2-error"][i] = np.Inf

  ##############################
  # Step 4: define pbar
  ##############################
  pbar = trange(params["num_rounds"])

  ##############################
  # Step 5: main traning loop
  ##############################
  for k_round in pbar:
    # define local weights and losses
    if params["sharing_mode"] == "all":
      local["weights"] = []
    else:
      local["branch_weights"] = []
      local["trunk_weights"] = []
    local["losses"] = []

    for i in range(num_clients):
      models[i].train()
      local_model = DON_local_update(
        params,
        dataset=(u_train, y_train, G_train),
        idxs=client_groups,
        logger=logger,
        client_number=i,
        device=device,
      )

      # local update of weights
      if params["sharing_mode"] == "all":
        w, loss, num_grandients = local_model.update_weights(
          model=copy.deepcopy(models[i]), 
          global_round=k_round+1
          )
        local["weights"].append(copy.deepcopy(w))

      else:
        w_branch, w_trunk, loss, num_gradients = local_model.update_weights(
          model=copy.deepcopy(models[i]), 
          global_round=k_round+1, 
          sharing_mode=params["sharing_mode"]
          )
        local["branch_weights"].append(copy.deepcopy(w_branch))
        local["trunk_weights"].append(copy.deepcopy(w_trunk))
      logger["total_gradients"] += num_gradients
      local["losses"].append(copy.deepcopy(loss))

    # synchronize selected weights
    if params["sharing_mode"] == "all":
      global_weights = average_weights(local["weights"])
    elif params["sharing_mode"] == "only_branch":
      global_branch_weights = average_weights(local["branch_weights"])
    elif params["sharing_mode"] == "only_trunk":
      global_trunk_weights = average_weights(local["trunk_weights"])
    
    # update models
    for i in range(num_clients):
      if params["sharing_mode"] == "all":
        models[i].load_state_dict(global_weights)
      elif params["sharing_mode"] == "only_branch":
        models[i].branch.load_state_dict(global_branch_weights)
        models[i].trunk.load_state_dict(local["trunk_weights"][i])
      elif params["sharing_mode"] == "only_trunk":
        models[i].branch.load_state_dict(local["branch_weights"][i])
        models[i].trunk.load_state_dict(global_trunk_weights)
      else:
        models[i].branch.load_state_dict(local["branch_weights"][i])
        models[i].trunk.load_state_dict(local["trunk_weights"][i])

    # compute average loss
    loss_avg = sum(local["losses"]) / len(local["losses"])

    # append training loss
    logger["train_loss"].append(loss_avg)

    # update best models
    for i in range(num_clients):
      if local["losses"][i] < logger["best_loss"][i]:
        logger["best_loss"][i] = local["losses"][i]
        save_file = save_name + str(i) + ".pt"
        save(models[i], save_path=save_file)

    if loss_avg < logger["best_global_loss"]:
      logger["best_global_loss"] = loss_avg 

    # print updates
    pbar.set_postfix(
      {
        'Local losses': np.round(local["losses"],6),
        'Local best losses': logger["best_loss"],
        'Global training loss': np.round(loss_avg,6),
        'Global best loss:': np.round(logger["best_global_loss"],6),
        'number of grads': np.round(logger["total_gradients"],6),
        }
        )
    
  total_gradients = np.round(logger["total_gradients"],6)
  print(f"\n******** training summary ********\n")
  print(f'model client-1 best loss ={np.round(logger["best_loss"][0],6)}')
  print(f'model client-2 best loss ={np.round(logger["best_loss"][1],6)}')
  print(f'\nthe number of gradients computed is={total_gradients}')
  return logger
  
##############################
# Function: D2NO train and test function
##############################
def D2NO_train_and_test(
  models: Dict,
  dataset: Tuple,
  num_clients: int,
  params: Dict,
  client_groups: Dict,
  test_dataset: Tuple,
  num_pred: int,
  means: Tuple,
  st_devs: Tuple,
  num_test: int= 1,
  device: torch.device = torch.device("cpu")
  ) -> Dict:
  ##############################
  # Step 1: unpack the dataset
  ##############################
  u_train, y_train, G_train = dataset 
  u_test, y_test, G_test = test_dataset

  ##############################
  # Step 2: copy weights
  ##############################
  global_weights = models[0].state_dict()

  ##############################
  # Step 3: define logger
  ##############################
  logger = {}
  local= {}
  logger["train_loss"] = []
  
  logger["best_loss"] = {} 
  logger["best_l2_error"] = {}
  for i in range(num_clients):
    logger["best_loss"][i] = np.Inf
    logger["best_l2_error"][i] = np.Inf
  logger["best_global_loss"] = np.Inf
  

  ##############################
  # Step 4: define pbar
  ##############################
  pbar = trange(params["num_rounds"])

  ##############################
  # Step 5: main traning loop
  ##############################
  for k_round in pbar:
    # define local weights and losses
    if params["sharing_mode"] == "all":
      local["weights"] = []
    else:
      local["branch_weights"] = []
      local["trunk_weights"] = []
    local["losses"] = []

    if k_round * params["local_epochs"] == 40:
      params["lr"] = params["gamma"] * params["lr"]
    elif k_round * params["local_epochs"] == 80:
      params["lr"] = params["gamma"] * params["lr"]
    elif k_round * params["local_epochs"] == 120:
      params["lr"] = params["gamma"] * params["lr"]

    for i in range(num_clients):
      models[i].train()
      local_model = DON_local_update(
        params,
        dataset=(u_train, y_train, G_train),
        idxs=client_groups,
        logger=logger,
        client_number=i,
        device=device,
      )
      # local update of weights
      if params["sharing_mode"] == "all":
        w, loss = local_model.update_weights(
          model=copy.deepcopy(models[i]), 
          global_round=k_round+1
          )
        local["weights"].append(copy.deepcopy(w))

      else:
        w_branch, w_trunk, loss = local_model.update_weights(
          model=copy.deepcopy(models[i]), 
          global_round=k_round+1, 
          sharing_mode=params["sharing_mode"]
          )
        local["branch_weights"].append(copy.deepcopy(w_branch))
        local["trunk_weights"].append(copy.deepcopy(w_trunk))

      local["losses"].append(copy.deepcopy(loss))

    # synchronize selected weights
    if params["sharing_mode"] == "all":
      global_weights = average_weights(local["weights"])
    elif params["sharing_mode"] == "only_branch":
      global_branch_weights = average_weights(local["branch_weights"])
    elif params["sharing_mode"] == "only_trunk":
      global_trunk_weights = average_weights(local["trunk_weights"])
    
    # update models
    for i in range(num_clients):
      if params["sharing_mode"] == "all":
        models[i].load_state_dict(global_weights)
      elif params["sharing_mode"] == "only_branch":
        models[i].branch.load_state_dict(global_branch_weights)
        models[i].trunk.load_state_dict(local["trunk_weights"][i])
      elif params["sharing_mode"] == "only_trunk":
        models[i].branch.load_state_dict(local["branch_weights"][i])
        models[i].trunk.load_state_dict(global_trunk_weights)
      else:
        models[i].branch.load_state_dict(local["branch_weights"][i])
        models[i].trunk.load_state_dict(local["trunk_weights"][i])

    # compute average loss
    loss_avg = sum(local["losses"]) / len(local["losses"])

    # append training loss
    logger["train_loss"].append(loss_avg)

    # update best models
    for i in range(num_clients):
      if local["losses"][i] < logger["best_loss"][i]:
        logger["best_loss"][i] = local["losses"][i]
        save_file = "./best-model-" + str(i) + ".pt"
        save(models[i], save_path=save_file)

    if loss_avg < logger["best_global_loss"]:
      logger["best_global_loss"] = loss_avg 

    # test models
    l2_error = {}
    l2_error[0] = []
    l2_error[1] = []

    for k in range(num_test):
      with torch.no_grad():
        for i in range(num_clients):
          u_test_example = u_test[i][k*num_pred:(k+1)*num_pred,...].to(device)
          y_test_example = y_test[i][k*num_pred:(k+1)*num_pred,...].to(device)
          G_test_example = G_test[i][k*num_pred:(k+1)*num_pred,...]
          models[i].eval()
          # perform inference
          G_test_pred = models[i]((u_test_example, y_test_example))
          l2_error[i].append(
            compute_l2_error(
              G_test_example.flatten(), 
              unnormalize(G_test_pred.cpu().detach().numpy(), means[i], st_devs[i]).flatten()
              )
              )
    
    avg_l2_error = {}
    avg_l2_error[0] = np.mean(l2_error[0])
    avg_l2_error[1] = np.mean(l2_error[1])

    # check best l2-errors
    for i in range(num_clients):
      logger["best_l2_error"][i] = avg_l2_error[i] if avg_l2_error[i] < logger["best_l2_error"][i] else logger["best_l2_error"][i]  

    # print updates
    pbar.set_postfix(
      {
        'Local losses': local["losses"],
#        'Local best losses': logger["best_loss"],
#        'Global training loss': loss_avg,
#        'Global best loss:': logger["best_global_loss"],
#        'local l2-errors': avg_l2_error,
        'best l2-errors': logger["best_l2_error"]
        }
        )
    
  return logger


