# Standard imports
import copy
import numpy as np

from tqdm.auto import trange
from typing import Any, Dict, List, Tuple

# Deep learning imports
import torch

# My imports 
from models.operator import DON_local_update
from utils.utils import compute_l2_error, unnormalize, save

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

      print(f'current number of total_gradients taken is = {logger["total_gradients"]}')
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
        'Local losses': local["losses"],
        'Local best losses': logger["best_loss"],
        'Global training loss': loss_avg,
        'Global best loss:': logger["best_global_loss"]
        }
        )
    
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


