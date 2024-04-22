##############################
# standard imports
##############################
import argparse
import numpy as np

##############################
# deep learning imports
##############################
import torch

##############################
# my imports
##############################
from data.data_structures import Dataset, get_FWI_client_data
from models.operator import classical_DON, old_classical_DON
from supervisor.D2NO_training import D2NO_train_and_test
from utils.utils import get_num_params, normalize, set_seed

##############################
# config vars
##############################
SEED = 1234
VERBOSE = False
USE_OLD = False
NORMALIZE = True
dim = 2
sensor = 1

##############################
# Distributed hyperparams
##############################
DISTRIBUTED_HYPERPARAMS = {"num_clients": 2}

##############################
# Data hyperparams
##############################
DATA_HYPERPARAMS = {
	"input_sub_sample_client_1": 2, # 3
	"input_sub_sample_client_2": 2, # 2
	"output_time_sub_sample": 5, # 5
    	"output_offset_sub_sample": 7, # 10
    	"different_num_sensors": True, 
    	"num_test": 100,
    }

def main(args):
	##############################
	# step 1: define the model hyperparams
	##############################		
	DATA_HYPERPARAMS["num_experiments"] = args.num_experiments
	MODEL_HYPERPARAMS = {
    		"branch_type": args.branch_type, # modified
    		"trunk_type": args.trunk_type, 
    		"branch_act_fn_name": args.branch_act_fn_name, 
    		"branch_width": args.branch_width,
    		"branch_depth": {0: args.branch_depth_0, 1: args.branch_depth_1},
    		"trunk_act_fn_name": args.trunk_act_fn_name,
    		"trunk_width": args.trunk_width, 
	    	"trunk_depth": args.trunk_depth,
    		"num_basis": args.num_basis,
	}

	##############################	
	# step 2: define the learning hyperparams
	##############################
	LEARNING_HYPERPARAMS = {}

	LEARNING_HYPERPARAMS["num_rounds"] = args.num_rounds
	LEARNING_HYPERPARAMS["sharing_mode"] = "only_trunk"
	LEARNING_HYPERPARAMS["different_num_sensors"] = DATA_HYPERPARAMS["different_num_sensors"]
	LEARNING_HYPERPARAMS["batch_size"] = 1024
	LEARNING_HYPERPARAMS["optimizer"] = 'adam'
	LEARNING_HYPERPARAMS["lr"] = args.lr
	LEARNING_HYPERPARAMS["local_epochs"] = args.local_epochs
	LEARNING_HYPERPARAMS["verbose_local"] = False
	LEARNING_HYPERPARAMS["gamma"] = args.gamma
	##############################
	# step 3: set RNG
	##############################
	set_seed(SEED)

	##############################
	# step 4: define device
	##############################
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	if VERBOSE:
		print(f"We are using: {device}")

	##############################
	# step 5: read the training and test datasets
	##############################
	
	# training data
	
	# input client 1
	input_data_for_training_client_1_1 = np.load('./data/FWI/FlatVelA/model1.npy')[:-40,...].squeeze()
	input_data_for_training_client_1_2 = np.load('./data/FWI/FlatVelA/model2.npy')[:-40,...].squeeze()
	input_data_for_training_client_1_3 = np.load('./data/FWI/FlatVelA/model3.npy')[:-40,...].squeeze()

	input_data_client_1 = np.vstack((input_data_for_training_client_1_1, input_data_for_training_client_1_2, input_data_for_training_client_1_3))[:DATA_HYPERPARAMS["num_experiments"],...]

	# input client 2
	input_data_for_training_client_2_1 = np.load('./data/FWI/FlatVelB/model1.npy')[:-40,...].squeeze()
	input_data_for_training_client_2_2 = np.load('./data/FWI/FlatVelB/model2.npy')[:-40,...].squeeze()
	input_data_for_training_client_2_3 = np.load('./data/FWI/FlatVelB/model3.npy')[:-40,...].squeeze()

	input_data_client_2 = np.vstack((input_data_for_training_client_2_1, input_data_for_training_client_2_2, input_data_for_training_client_2_3))[:DATA_HYPERPARAMS["num_experiments"],...]

	# output client 1
	output_data_for_training_client_1_1 = np.load('./data/FWI/FlatVelA/data1.npy')[:-40,sensor,...]
	output_data_for_training_client_1_2 = np.load('./data/FWI/FlatVelA/data2.npy')[:-40,sensor,...]
	output_data_for_training_client_1_3 = np.load('./data/FWI/FlatVelA/data3.npy')[:-40,sensor,...]
	
	output_data_client_1 = np.vstack((output_data_for_training_client_1_1, output_data_for_training_client_1_2, output_data_for_training_client_1_3))[:DATA_HYPERPARAMS["num_experiments"],...]

	# output client 2
	output_data_for_training_client_2_1 = np.load('./data/FWI/FlatVelB/data1.npy')[:-40,sensor,...]
	output_data_for_training_client_2_2 = np.load('./data/FWI/FlatVelB/data2.npy')[:-40,sensor,...]
	output_data_for_training_client_2_3 = np.load('./data/FWI/FlatVelB/data3.npy')[:-40,sensor,...]

	output_data_client_2 = np.vstack((output_data_for_training_client_2_1, output_data_for_training_client_2_2, output_data_for_training_client_2_3))[:DATA_HYPERPARAMS["num_experiments"],...]

	# test data

	# input client 1
	input_data_for_test_client_1_1 = np.load('./data/FWI/FlatVelA/model1.npy')[-40:,...].squeeze()
	input_data_for_test_client_1_2 = np.load('./data/FWI/FlatVelA/model2.npy')[-40:,...].squeeze()
	input_data_for_test_client_1_3 = np.load('./data/FWI/FlatVelA/model3.npy')[-40:,...].squeeze()

	input_test_data_client_1 = np.vstack((input_data_for_test_client_1_1, input_data_for_test_client_1_2, input_data_for_test_client_1_3))

	# input client 2
	input_data_for_test_client_2_1 = np.load('./data/FWI/FlatVelB/model1.npy')[-40:,...].squeeze()
	input_data_for_test_client_2_2 = np.load('./data/FWI/FlatVelB/model2.npy')[-40:,...].squeeze()
	input_data_for_test_client_2_3 = np.load('./data/FWI/FlatVelB/model3.npy')[-40:,...].squeeze()

	input_test_data_client_2 = np.vstack((input_data_for_test_client_2_1, input_data_for_test_client_2_2, input_data_for_test_client_2_3))

	# output client 1
	output_data_for_test_client_1_1 = np.load('./data/FWI/FlatVelA/data1.npy')[-40:,sensor,...]
	output_data_for_test_client_1_2 = np.load('./data/FWI/FlatVelA/data2.npy')[-40:,sensor,...]
	output_data_for_test_client_1_3 = np.load('./data/FWI/FlatVelA/data3.npy')[-40:,sensor,...]

	output_test_data_client_1 = np.vstack((output_data_for_test_client_1_1, output_data_for_test_client_1_2, output_data_for_test_client_1_3))

	# output client 2
	output_data_for_test_client_2_1 = np.load('./data/FWI/FlatVelB/data1.npy')[-40:,sensor,...]
	output_data_for_test_client_2_2 = np.load('./data/FWI/FlatVelB/data2.npy')[-40:,sensor,...]
	output_data_for_test_client_2_3 = np.load('./data/FWI/FlatVelB/data3.npy')[-40:,sensor,...]

	output_test_data_client_2 = np.vstack((output_data_for_test_client_2_1, output_data_for_test_client_2_2, output_data_for_test_client_2_3))

	
	if VERBOSE:
		print(f"The input training data shape for the forward problem of client 1 is {input_data_client_1.shape}")
		print(f"The output training data shape for the forward problem of client 1 is {output_data_client_1.shape}")
		print(f"The input training data shape for the forward problem of client 2 is {input_data_client_2.shape}")
		print(f"The output training data shape for the forward problem of client 2 is {output_data_client_2.shape}")

		print(f"\nThe input test data shape for the forward problem of client 1 is {input_test_data_client_1.shape}")
		print(f"The output test data shape for the forward problem of client 1 is {output_test_data_client_1.shape}")
		print(f"The input test data shape for the forward problem of client 2 is {input_test_data_client_2.shape}")	
		print(f"The output test data shape for the forward problem of client 2 is {output_test_data_client_2.shape}")

	##############################
	# step 6: define the one dimensional input for D2NO clients
	##############################
	u_client_1 = input_data_client_1[:,::DATA_HYPERPARAMS["input_sub_sample_client_1"],0]
	u_client_2 = input_data_client_2[:,::DATA_HYPERPARAMS["input_sub_sample_client_2"],0]

	# test inputs 
	u_test_client_1 = input_test_data_client_1[:,::DATA_HYPERPARAMS["input_sub_sample_client_1"],0]
	u_test_client_2 = input_test_data_client_2[:,::DATA_HYPERPARAMS["input_sub_sample_client_2"],0]

	if VERBOSE:
		print(f"The shape of the one-dim input for client 1 is {u_client_1.shape}")
		print(f"The shape of the one-dim input for client 2 is {u_client_2.shape}")
	
		print(f"\nThe shape of the one-dim test input for client 1 is {u_test_client_1.shape}")
		print(f"The shape of the one-dim test input for client 2 is {u_test_client_2.shape}")

	##############################
	# step 7: define the output for D2NO clients
	##############################
	G_client_1 = output_data_client_1[:,::DATA_HYPERPARAMS["output_time_sub_sample"], ::DATA_HYPERPARAMS["output_offset_sub_sample"]]
	G_client_2 = output_data_client_2[:,::DATA_HYPERPARAMS["output_time_sub_sample"], ::DATA_HYPERPARAMS["output_offset_sub_sample"]]

	# test outputs
	G_test_client_1 = output_test_data_client_1[:,::DATA_HYPERPARAMS["output_time_sub_sample"], ::DATA_HYPERPARAMS["output_offset_sub_sample"]]
	G_test_client_2 = output_test_data_client_2[:,::DATA_HYPERPARAMS["output_time_sub_sample"], ::DATA_HYPERPARAMS["output_offset_sub_sample"]]

	num_pred = G_client_1.shape[1] * G_client_1.shape[-1]

	if VERBOSE:
		print(f"The shape for client 1 output data is {G_client_1.shape}")
		print(f"The shape for client 2 output data is {G_client_2.shape}")
		print(f"\nThe shape for client 1 output test data is {G_test_client_1.shape}")
		print(f"The shape for client 2 output test data is {G_test_client_2.shape}")

	##############################
	# step 8: create DON data for clients
	##############################
	ts = np.arange(0.0, 1.0, step = 1.0/G_client_1.shape[1]).flatten()
	offsets = np.arange(0.0, 1.0, step = 1.0/G_client_1.shape[-1]).flatten()

	u_train_client_1, y_train_client_1, G_train_client_1 = get_FWI_client_data(
		u_client_1,
		G_client_1,
		ts,
		offsets
	)

	u_train_client_2, y_train_client_2, G_train_client_2 = get_FWI_client_data(
		u_client_2,
		G_client_2,
		ts,
		offsets
	)

	# test data 
	u_test_client_1, y_test_client_1, G_test_client_1 = get_FWI_client_data(
		u_test_client_1,
		G_test_client_1,
		ts,
		offsets
	)

	u_test_client_2, y_test_client_2, G_test_client_2 = get_FWI_client_data(
		u_test_client_2,
		G_test_client_2,
		ts,
		offsets
	)

	if NORMALIZE:
		client_1_dataset = Dataset(u_train_client_1, y_train_client_1, G_train_client_1)
		client_2_dataset = Dataset(u_train_client_2, y_train_client_2, G_train_client_2)

		client_1_dataset.update_statistics()
		client_2_dataset.update_statistics()

		u_train_client_1 = normalize(client_1_dataset.dataU, client_1_dataset.dataU_mean, client_1_dataset.dataU_std)
		u_train_client_2 = normalize(client_2_dataset.dataU, client_2_dataset.dataU_mean, client_2_dataset.dataU_std)

		G_train_client_1 = normalize(client_1_dataset.dataG, client_1_dataset.dataG_mean, client_1_dataset.dataG_std)
		G_train_client_2 = normalize(client_2_dataset.dataG, client_2_dataset.dataG_mean, client_2_dataset.dataG_std)

		# test data
		client_1_test_dataset = Dataset(u_test_client_1, y_test_client_1, G_test_client_1)
		client_2_test_dataset = Dataset(u_test_client_2, y_test_client_2, G_test_client_2)

		u_test_client_1 = normalize(client_1_test_dataset.dataU, client_1_dataset.dataU_mean, client_1_dataset.dataU_std)
		u_test_client_2 = normalize(client_2_test_dataset.dataU, client_2_dataset.dataU_mean, client_2_dataset.dataU_std)

	if VERBOSE:
		print(f"The shape of client 1 training data is: u={u_train_client_1.shape}, y={y_train_client_1.shape}, G={G_train_client_1.shape}")
		print(f"The shape of client 2 training data is: u={u_train_client_2.shape}, y={y_train_client_2.shape}, G={G_train_client_2.shape}")

		print(f"\nThe shape of client 1 test data is: u={u_test_client_1.shape}, y={y_test_client_1.shape}, G={G_test_client_1.shape}")
		print(f"The shape of client 2 test data is: u={u_test_client_2.shape}, y={y_test_client_2.shape}, G={G_test_client_2.shape}")

	##############################
	# step 9: collect the dataset
	##############################
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
		
	##############################
	# step 10: build the DONs
	##############################
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

	##############################
	# step 11: print the number of params
	##############################
	if VERBOSE:
		for i in range(DISTRIBUTED_HYPERPARAMS["num_clients"]):
			print(
				f"Client {i+1} number of branch params = {get_num_params(models[i].branch)} and trunk params = {get_num_params(models[i].trunk)}"
			)

	##############################
	# step 12: D2NO training
	##############################
	dataset=(u_train_torch, y_train_torch, G_train_torch)

	# test data
	test_dataset=(u_test_torch, y_test_torch, G_test)
	num_test = 120
	means = (client_1_dataset.dataG_mean, client_2_dataset.dataG_mean)
	st_devs = (client_1_dataset.dataG_std, client_2_dataset.dataG_std)

	_ = D2NO_train_and_test(
		models,
		dataset,
		DISTRIBUTED_HYPERPARAMS["num_clients"],
		LEARNING_HYPERPARAMS,
		client_groups=client_groups,
		test_dataset=test_dataset,
		num_pred=num_pred,
		means=means,
		st_devs=st_devs,
		num_test=num_test,
		device=device,
		)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="D2NO")
   	
	parser.add_argument('--num-rounds', type=int, default=60, help="number of training rounds")
	parser.add_argument('--num-experiments', type=int, default=900, help="number of experiments")
	parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")	
	parser.add_argument('--local-epochs', type=int, default=1, help="number of local training steps")
	parser.add_argument('--branch-type', type=str, default="modified", help="branch network type")
	parser.add_argument('--trunk-type', type=str, default="FNN", help="trunk network type")
	parser.add_argument('--branch-act-fn-name', type=str, default="leaky", help="branch activation function name")
	parser.add_argument('--trunk-act-fn-name', type=str, default="leaky", help="trunk activation function name")
	parser.add_argument('--branch-width', type=int, default=100, help="width of branch network")
	parser.add_argument('--trunk-width', type=int, default=100, help="width of trunk network")
	parser.add_argument('--branch-depth-0', type=int, default=2, help="depth of branch network")
	parser.add_argument('--branch-depth-1', type=int, default=2, help="depth of branch network")
	parser.add_argument('--trunk-depth', type=int, default=3, help="depth of trunk network")
	parser.add_argument('--num-basis', type=int, default=100, help="number of basis")
	parser.add_argument('--gamma', type=float, default=0.5, help="multiplicative factor for lr")	
	args = parser.parse_args()
	main(args)


