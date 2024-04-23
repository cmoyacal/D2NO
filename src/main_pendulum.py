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
from data.data_structures import get_distributed_dataset
from data.fun_spaces import GRF
from data.ode_data import ode_system
from models.operator import classical_DON, old_classical_DON
from supervisor.D2NO_inference import local_inference
from supervisor.D2NO_training import D2NO_train
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
    DATA_HYPERPARAMS["different_num_sensors"] = True
    DATA_HYPERPARAMS["num_train"] = args.num_train
    DATA_HYPERPARAMS["different_num_train"] = False     # check if the code holds with this = True

    # check the len of num_sensors and num_train are the same
    assert len(DATA_HYPERPARAMS["num_sensors"]) == len(DATA_HYPERPARAMS["num_train"])

    #################################
    # define the learning hyperparams
    #################################
    LEARNING_HYPERPARAMS = {}

    LEARNING_HYPERPARAMS["num_rounds"] = args.num_rounds
    LEARNING_HYPERPARAMS["sharing_mode"] = "only_trunk"
    LEARNING_HYPERPARAMS["different_num_sensors"] = DATA_HYPERPARAMS["different_num_sensors"]
    LEARNING_HYPERPARAMS["batch_size"] = 64
    LEARNING_HYPERPARAMS["optimizer"] = 'adam'
    LEARNING_HYPERPARAMS["lr"] = args.lr
    LEARNING_HYPERPARAMS["local_epochs"] = args.local_epochs
    LEARNING_HYPERPARAMS["verbose_local"] = False

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
        u_train_dict[i], y_train_dict[i], G_train_dict[i] = system.gen_operator_data(spaces[i], DATA_HYPERPARAMS["num_sensors"][i], clients[i]["num_train"], verbose=False)

    if not DATA_HYPERPARAMS["different_num_sensors"] and not DATA_HYPERPARAMS["different_num_train"]:
        u_train = np.vstack([u_train_dict[i] for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])])
        y_train = np.vstack([y_train_dict[i] for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])])
        G_train = np.vstack([G_train_dict[i] for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])])

        # move to torch (we will move to device later)
        u_train_torch = torch.tensor(u_train, dtype=torch.float32, device=torch.device("cpu"))
        y_train_torch = torch.tensor(y_train, dtype=torch.float32, device=torch.device("cpu"))
        G_train_torch = torch.tensor(G_train, dtype=torch.float32, device=torch.device("cpu"))

        if VERBOSE:
            for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
                print(f"The size of the {i+1}th client is u={u_train_dict[i].shape}, y={y_train_dict[i].shape}, and G={G_train_dict[i].shape}")
            print(f"\nFull dataset size is u={u_train_torch.shape}, y={y_train_torch.shape}, and G={G_train_torch.shape}")
    
        client_groups = get_distributed_dataset(G_train, DISTRIBUTED_HYPERPARAMS["num_clients"], verbose=VERBOSE)

    else:
        u_train = [u_train_dict[i] for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]
        y_train = [y_train_dict[i] for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]
        G_train = [G_train_dict[i] for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]

        # move to torch (we will move to device later)
        u_train_torch = [torch.tensor(u_train[i], dtype=torch.float32, device=torch.device("cpu")) for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]
        y_train_torch = [torch.tensor(y_train[i], dtype=torch.float32, device=torch.device("cpu")) for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]
        G_train_torch = [torch.tensor(G_train[i], dtype=torch.float32, device=torch.device("cpu")) for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]

        client_groups = {}

        if VERBOSE:
            for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
                print(f"The size of {i+1}th client's dataset is u={u_train[i].shape}, y={y_train[i].shape}, and G={G_train[i].shape}")

    ################
    # build the DONs
    ################
    models = {}
    for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
        branch = {}
        branch["type"] = MODEL_HYPERPARAMS["branch_type"]
        branch["act_fn_name"] = MODEL_HYPERPARAMS["act_fn_name"]
        branch["layer_size"] = [DATA_HYPERPARAMS["num_sensors"][i]] + [MODEL_HYPERPARAMS["width"]] * MODEL_HYPERPARAMS["depth"] + [MODEL_HYPERPARAMS["num_basis"]]

        trunk = {}
        trunk["type"] = MODEL_HYPERPARAMS["trunk_type"]
        trunk["act_fn_name"] = MODEL_HYPERPARAMS["act_fn_name"]
        trunk["layer_size"] = [dim] + [MODEL_HYPERPARAMS["width"]] * MODEL_HYPERPARAMS["depth"] + [MODEL_HYPERPARAMS["num_basis"]]

        if not USE_OLD:
            models[i] = classical_DON(branch, trunk, use_bias=False).to(device)
        else:
            models[i] = old_classical_DON(branch, trunk, use_bias=False).to(device)
        del branch, trunk

        if VERBOSE:
            print(models[i])

    ############################
    # train the number of params
    ############################
    if VERBOSE:
        for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
            print(f"Client {i+1} number of branch params = {get_num_params(models[i].branch)} and trunk params = {get_num_params(models[i].trunk)}")

    ##############
    # D2NO traning
    ##############
    dataset=(u_train_torch, y_train_torch, G_train_torch)
    save_name="./output/pendulum-best-model-"

    _ = D2NO_train(
        models,
        dataset,
        DISTRIBUTED_HYPERPARAMS["num_clients"],
        LEARNING_HYPERPARAMS,
        client_groups=client_groups,
        device=device,
        save_name=save_name
    )

    ## TESTING ##

    ###################################################
    # recover the best models from distributed training
    ###################################################
    for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
        ckpt_file = save_name + str(i) + ".pt"
        ckpt = restore(ckpt_file)
        state_dict = ckpt['state_dict']
        models[i].load_state_dict(state_dict)
        models[i].to(device)
        del ckpt_file, ckpt, state_dict

    ################
    # test the model
    ################
    for i in range(DISTRIBUTED_HYPERPARAMS['num_clients']):
        errors = local_inference(
            models,
            system,
            spaces,
            num_test=100,
            num_sensors=DATA_HYPERPARAMS["num_sensors"],
            client_number=i,
            T=DATA_HYPERPARAMS["T"],
            device=device,
            idxs_plot=None,
            verbose=VERBOSE,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D2NO-pendulum")

    # distributed 
    parser.add_argument('--num-clients', type=int, default=2, help="number of clients")
    parser.add_argument('--num-sensors', nargs="+", type=int, default=[100, 10], help="list of number of sensors")
    parser.add_argument('--num-train', nargs="+", type=int, default=[1000, 1000], help="list of number of training samples")

    # learning
    parser.add_argument('--num-rounds', type=int, default=60, help="number of training rounds")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")	
    parser.add_argument('--local-epochs', type=int, default=1, help="number of local training steps")

    args = parser.parse_args()
    main(args)