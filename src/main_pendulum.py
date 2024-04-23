##########
# standard
##########
import argparse
import numpy as np

###############
# deep learning
###############
import torch

############
# my imports 
############
from data.data_structures import OperatorDataset
from data.fun_spaces import GRF
from data.ode_data import ode_system
from models.operator import classical_DON, old_classical_DON
from supervisor.D2NO_inference import centralized_inference
from supervisor.D2NO_training import centralized_train
from utils.utils import get_num_params, restore, set_seed

##################
# config variables
##################
SEED = 1234
VERBOSE = True
USE_OLD = False
dim = 1

min_ell = 0.1
max_ell = 1.0

def main(args):
     ##########
    # set seed
    ##########
    set_seed(SEED)

    ###############
    # define device
    ###############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ################################
    # define distributed hyperparams
    ################################
    DISTRIBUTED_HYPERPARAMS = {"num_clients": args.num_clients}

    #########################
    # define data hyperparams
    #########################
    DATA_HYPERPARAMS = {}
    DATA_HYPERPARAMS["length_scale"] = np.linspace(min_ell, max_ell, num=DISTRIBUTED_HYPERPARAMS["num_clients"]).flatten()
    DATA_HYPERPARAMS["T"] = 1.0
    DATA_HYPERPARAMS["num_sensors"] = args.num_sensors
    DATA_HYPERPARAMS["num_train"] = args.num_train
    DATA_HYPERPARAMS["different_num_train"] = False     # check if the code holds with this = True

    #################################
    # define the learning hyperparams
    #################################
    LEARNING_HYPERPARAMS = {}

    LEARNING_HYPERPARAMS["num_epochs"] = args.num_epochs
    LEARNING_HYPERPARAMS["batch_size"] = 64
    LEARNING_HYPERPARAMS["optimizer"] = 'adam'
    LEARNING_HYPERPARAMS["lr"] = args.lr
    
    ###################
    # model hyperparams
    ###################
    MODEL_HYPERPARAMS = {
        "branch_type": "FNN",
        "trunk_type": "FNN",
        "act_fn_name": "relu",
        "width": 50,
        "depth": 1,
        "num_basis": 50,
    }

    ################
    # create clients
    ################
    clients = {}
    for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
        clients[i] = {}
        clients[i]["num_train"] = DATA_HYPERPARAMS['num_train'][i]
        clients[i]["length_scale"] = DATA_HYPERPARAMS['length_scale'][i]

    if VERBOSE:
        print(f"Clients' details are:\n{clients}")

    ################
    # create dataset
    ################
    system = ode_system(DATA_HYPERPARAMS["T"])
    spaces = {}
    for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
        spaces[i] = GRF(T=DATA_HYPERPARAMS['T'], length_scale=clients[i]["length_scale"], N=1000, interp="cubic")

    # create data dicts
    u_train_dict = {}
    y_train_dict = {}
    G_train_dict = {}

    for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
        u_train_dict[i], y_train_dict[i], G_train_dict[i] = system.gen_operator_data(spaces[i], DATA_HYPERPARAMS["num_sensors"], clients[i]["num_train"], verbose=False)

    # this code handles only same num_train. For different, see distributed
    u_train_full = np.vstack([u_train_dict[i] for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])])
    y_train_full = np.vstack([y_train_dict[i] for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])])
    G_train_full = np.vstack([G_train_dict[i] for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])])

    # move the data to torch - keep it in cpu before training
    u_train = torch.tensor(u_train_full, dtype=torch.float32, device=torch.device("cpu"))
    y_train = torch.tensor(y_train_full, dtype=torch.float32, device=torch.device("cpu"))
    G_train = torch.tensor(G_train_full, dtype=torch.float32, device=torch.device("cpu"))

    if VERBOSE:
        print(f"Full dataset size is u={u_train.shape}, y={y_train.shape}, and G={G_train.shape}")

    #########################
    # construct torch dataset
    #########################
    torch_dataset = OperatorDataset(u_train, y_train, G_train)

    #######################
    # build centralized DON
    #######################
    branch = {}
    branch["type"] = MODEL_HYPERPARAMS["branch_type"]
    branch["act_fn_name"] = MODEL_HYPERPARAMS["act_fn_name"]
    branch["layer_size"] = [DATA_HYPERPARAMS["num_sensors"][i]] + [MODEL_HYPERPARAMS["width"]] * MODEL_HYPERPARAMS["depth"] + [MODEL_HYPERPARAMS["num_basis"]]

    trunk = {}
    trunk["type"] = MODEL_HYPERPARAMS["trunk_type"]
    trunk["act_fn_name"] = MODEL_HYPERPARAMS["act_fn_name"]
    trunk["layer_size"] = [dim] + [MODEL_HYPERPARAMS["width"]] * MODEL_HYPERPARAMS["depth"] + [MODEL_HYPERPARAMS["num_basis"]]

    if not USE_OLD:
        centralized_model = classical_DON(branch, trunk, use_bias=False).to(device)
    else:
        centralized_model = old_classical_DON(branch, trunk, use_bias=False).to(device)

    if VERBOSE:
        print(centralized_model)

    #########################################
    # print the number of training parameters
    #########################################
    if VERBOSE:
        print(f"number of branch params = {get_num_params(centralized_model.branch)} and trunk params = {get_num_params(centralized_model.trunk)}\n")

    ##########################
    # run centralized training
    ##########################
    centralized_logger = centralized_train(
        model=centralized_model,
        dataset=torch_dataset,
        params=LEARNING_HYPERPARAMS,
        device=device
        )

    ####################
    # recover best model
    ####################
    ckpt = restore("./output/centralized-best-model.pt")
    state_dict = ckpt['state_dict']
    centralized_model.load_state_dict(state_dict)
    centralized_model.to(device)

    ###########################
    # run centralized inference
    ###########################
    for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
        errors = centralized_inference(
            model=centralized_model,
            system=system,
            spaces=spaces,
            client_number=i,
            num_test=args.num_test,
            num_sensors=DATA_HYPERPARAMS["num_sensors"],
            T=DATA_HYPERPARAMS["T"],
            device=device,
            verbose=VERBOSE,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D2NO-pendulum")

    # centralized 
    parser.add_argument('--num-clients', type=int, default=2, help="number of clients")
    parser.add_argument('--num-sensors', type=int, default=100, help="Number of sensors")
    parser.add_argument('--num-train', nargs="+", type=int, default=[1000, 1000], help="list of number of training samples")
    parser.add_argument('--num-test', type=int, default=100, help="number of test samples")

    # learning
    parser.add_argument('--num-epochs', type=int, default=60, help="number of training rounds")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")	

    args = parser.parse_args()
    main(args)