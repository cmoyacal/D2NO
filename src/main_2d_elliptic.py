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
from models.operator import classical_DON, old_classical_DON
from utils.utils import get_num_params, restore, set_seed
from supervisor.D2NO_inference import D2NO_inference
from supervisor.D2NO_training import D2NO_train

#############
# config vars
#############
SEED = 1234
VERBOSE = True
dim = 2
USE_OLD = False

DISTRIBUTED_HYPERPARAMS = {"num_clients": 2}

################
# function: main
################
def main(args):
    #####################
    # define model params
    #####################
    MODEL_HYPERPARAMS = {
        "branch_type": args.branch_type,
        "trunk_type": args.trunk_type,
        "branch_act_fn_name": args.branch_act_fn_name,
        "branch_width": args.branch_width,
        "branch_depth": {0: args.branch_depth_client_1, 1: args.branch_depth_client_2},
        "trunk_act_fn_name": args.trunk_act_fn_name,
        "trunk_width": args.trunk_width,
        "trunk_depth": args.trunk_depth,
        "num_basis": args.num_basis,
        }

    #############################
    # define learning hyperparams
    #############################
    LEARNING_HYPERPARAMS = {}

    LEARNING_HYPERPARAMS["num_rounds"] = args.num_rounds
    LEARNING_HYPERPARAMS["sharing_mode"] = "only_trunk"
    LEARNING_HYPERPARAMS["batch_size"] = 1024
    LEARNING_HYPERPARAMS["optimizer"] = 'adam'
    LEARNING_HYPERPARAMS["lr"] = args.lr
    LEARNING_HYPERPARAMS["verbose_local"] = False 
    LEARNING_HYPERPARAMS["different_num_sensors"] = True

    ##########
    # set seed
    ##########
    set_seed(SEED)

    ###############
    # define device
    ###############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if VERBOSE:
        print(f"We are using: {device}")

    ################################
    # get training data for client 1
    ################################
    with np.load(args.training_data_client1_filename) as data:
        u_full_train_client_1 = data["u"]
        u_red_train_client_1 = data["u_red"]
        y_train_client_1 = data["y"]
        G_train_client_1 = data["G"]

    ###############################
    # select the input for client 1
    ###############################
    u_train_client_1 = u_full_train_client_1 if not args.use_reduced_input else u_red_train_client_1

    ############################
    # get test data for client 1
    ############################
    with np.load(args.test_data_client1_filename) as data:
        u_full_test_client_1 = data["u"]
        u_red_test_client_1 = data["u_red"]
        y_test_client_1 = data["y"]
        G_test_client_1 = data["G"]

    ###############################
    # select the input for client 1
    ###############################
    u_test_client_1 = u_full_test_client_1 if not args.use_reduced_input else u_red_test_client_1
    
    ################################
    # get training data for client 2
    ################################
    with np.load(args.training_data_client2_fileneme) as data:
        u_full_train_client_2 = data["u"]
        u_red_train_client_2 = data["u_red"]
        y_train_client_2 = data["y"]
        G_train_client_2 = data["G"]

    ###############################
    # select the input for client 2
    ###############################
    u_train_client_2 = u_full_train_client_2 if not args.use_reduced_input else u_red_train_client_2

    ############################
    # get test data for client 2
    ############################
    with np.load(args.test_data_client2_fileneme) as data:
        u_full_test_client_2 = data["u"]
        u_red_test_client_2 = data["u_red"]
        y_test_client_2 = data["y"]
        G_test_client_2 = data["G"]

    ###############################
    # select the input for client 2
    ###############################
    u_test_client_2 = u_full_test_client_2 if not args.use_reduced_input else u_red_test_client_2
 
    #####################
    # collect the dataset
    #####################
    u_train = [u_train_client_1, u_train_client_2]
    y_train = [y_train_client_1, y_train_client_2]
    G_train = [G_train_client_1, G_train_client_2]

    # test data
    u_test = [u_test_client_1, u_test_client_2]
    y_test = [y_test_client_1, y_test_client_2]
    G_test = [G_test_client_1, G_test_client_2]

	# move to torch (we will move to device later)
    u_train_torch = [torch.tensor(u_train[i], dtype=torch.float32, device=torch.device("cpu")) for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]
    y_train_torch = [torch.tensor(y_train[i], dtype=torch.float32, device=torch.device("cpu")) for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]
    G_train_torch = [torch.tensor(G_train[i], dtype=torch.float32, device=torch.device("cpu")) for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]

	# move test input to torch (we will move to device later)
    u_test_torch = [torch.tensor(u_test[i], dtype=torch.float32, device=torch.device("cpu")) for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]
    y_test_torch = [torch.tensor(y_test[i], dtype=torch.float32, device=torch.device("cpu")) for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"])]

    client_groups = {}
    if VERBOSE:
        for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
            print(f"The size of {i+1}th client's dataset is u={u_train[i].shape}, y={y_train[i].shape}, and G={G_train[i].shape}")
            print(f"\nThe size of {i+1}th client's test dataset is u={u_test[i].shape}, y={y_test[i].shape}, and G={G_test[i].shape}")
		
    ################
    # build the DONS
    ################
    models = {}
    for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
        branch = {}
        branch["type"] = MODEL_HYPERPARAMS["branch_type"]
        branch["act_fn_name"] = MODEL_HYPERPARAMS["branch_act_fn_name"]
        branch["layer_size"] = [u_train[i].shape[-1]] + [MODEL_HYPERPARAMS["branch_width"]] * MODEL_HYPERPARAMS["branch_depth"][i] + [MODEL_HYPERPARAMS["num_basis"]]
        
        if MODEL_HYPERPARAMS["branch_type"] == "Fourier":
            branch["freqs"] = 4
            trunk = {}
            trunk["type"] = MODEL_HYPERPARAMS["trunk_type"]
            trunk["act_fn_name"] = MODEL_HYPERPARAMS["trunk_act_fn_name"]
            trunk["layer_size"] = [dim] + [MODEL_HYPERPARAMS["trunk_width"]] * MODEL_HYPERPARAMS["trunk_depth"] + [MODEL_HYPERPARAMS["num_basis"]]
        
        if not USE_OLD:
            models[i] = classical_DON(branch, trunk, use_bias=False).to(device)
        else:
            models[i] = old_classical_DON(branch, trunk, use_bias=False).to(device)
        del branch, trunk
        
        if VERBOSE:
            print(models[i])

        ############################
        # print the number of params
        ############################
        if VERBOSE:
            for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
                print(f"Client {i+1} number of branch params = {get_num_params(models[i].branch)} and trunk params = {get_num_params(models[i].trunk)}")

        ###############
        # D2NO training
        ###############
        dataset = (u_train_torch, y_train_torch, G_train_torch)

        _ = D2NO_train(
            models, 
            dataset,
            DISTRIBUTED_HYPERPARAMS["num_clients"],
            LEARNING_HYPERPARAMS,
            client_groups=client_groups,
            device=device,
        )

        ##############
        # D2NO testing
        ##############
        test_dataset = (u_test_torch, y_test_torch, G_test)
        
        #####################
        # restore best models
        #####################
        for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
            ckpt_file = "./output/best-model-" + str(i) + ".pt"
            ckpt = restore(ckpt_file)
            state_dict = ckpt['state_dict']
            models[i].load_state_dict(state_dict)
            models[i].to(device)
            del ckpt_file, ckpt, state_dict 


        for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
            _ = D2NO_inference(
                models,
                test_dataset,
                client_number=i,
                device=device,
                verbose=VERBOSE,
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D2NO-2d-elliptic")

    # model
    parser.add_argument('--branch-type', type=str, default="modified", help="branch network type")
    parser.add_argument('--trunk-type', type=str, default="FNN", help="trunk network type")
    parser.add_argument('--branch-act-fn-name', type=str, default="leaky", help="branch activation function name")
    parser.add_argument('--branch-width', type=int, default=100, help="width of branch network")
    parser.add_argument('--branch-depth-client-1', type=int, default=2, help="depth of branch network for client 1")
    parser.add_argument('--branch-depth-client-2', type=int, default=2, help="depth of branch network for client 2")
    parser.add_argument('--trunk-width', type=int, default=100, help="width of trunk network")
    parser.add_argument('--trunk-depth', type=int, default=3, help="depth of trunk network")
    parser.add_argument('--num-basis', type=int, default=100, help="number of basis")

    # learning
    parser.add_argument('--num-rounds', type=int, default=60, help="number of training rounds")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")	
    parser.add_argument('--local-epochs', type=int, default=1, help="number of local training steps")

    args = parser.parse_args()
    main(args)