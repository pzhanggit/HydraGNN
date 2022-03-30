import os, json
import matplotlib.pyplot as plt
from qm9_utils import *
import argparse
from tqdm import tqdm

############################################################################
# Parameters
############################################################################
parser = argparse.ArgumentParser()
# Input/Output
parser.add_argument(
    "--config",
    type=str,
    default="./qm9_gap.json",
    help="json config for qm9 in HydraGNN",
)
parser.add_argument(
    "--data_file",
    type=str,
    default="./dataset/gdb9_gap_cut.csv",
    help="file with [mol_id,smiles,homo,lumo,gap]",
)
parser.add_argument(
    "--data_split",
    type=str,
    default="./dataset/qm9_train_test_val_idx_lists.pkl",
    help="file with train/val/test splits",
)
parser.add_argument(
    "--output_directory",
    type=str,
    default="qm9_hydragnn",
    help="output directory and name for trained model",
)

args = parser.parse_args()
##################################################################################################################
input_filename = args.config
datafile_cut = args.data_file
trainvaltest_splitlists = args.data_split
log_name = (
    args.output_directory
)  # fixme: currently have to be in [./logs/<log_name>/<log_name>.pk]
##################################################################################################################
# read configuration file
##################################################################################################################
# fixme: will make graph_feature_names inferrable from data file
graph_feature_names = ["HOMO", "LUMO", "GAP"]  ##in eV
with open(input_filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]
var_config["output_names"] = [
    graph_feature_names[item] for ihead, item in enumerate(var_config["output_index"])
]
var_config["input_node_feature_names"] = node_attribute_names
var_config["graph_features_dim"] = [1]
var_config["node_feature_dim"] = [1 for var in node_attribute_names]
##################################################################################################################
# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()
##################################################################################################################
# load data and create loaders
##################################################################################################################
print("Loading data ...")
smiles_sets, values_sets, idxs_sets = datasets_load(
    datafile_cut, trainvaltest_splitlists
)
dataset_lists = [[] for dataset in values_sets]
for idataset, (smileset, valueset, idxset) in enumerate(
    zip(smiles_sets, values_sets, idxs_sets)
):
    for smilestr, ytarget, idx in tqdm(zip(smileset, valueset, idxset)):
        dataset_lists[idataset].append(
            generate_graphdata(idx, smilestr, ytarget, var_config)
        )
trainset = dataset_lists[0]
valset = dataset_lists[1]
testset = dataset_lists[2]

(
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
) = hydragnn.preprocess.create_dataloaders(
    trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
)
##################################################################################################################
# create model and optimizer
##################################################################################################################
config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"]["Architecture"],
    verbosity=verbosity,
)
model = hydragnn.utils.get_distributed_model(model, verbosity)

learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
)

hydragnn.utils.setup_log(log_name)
writer = hydragnn.utils.get_summary_writer(log_name)
with open("./logs/" + log_name + "/config.json", "w") as f:
    json.dump(config, f)
##################################################################################################################
# train model and save trained model
##################################################################################################################
hydragnn.train.train_validate_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    writer,
    scheduler,
    config["NeuralNetwork"],
    log_name,
    verbosity,
    config["Visualization"]["plot_init_solution"],
    config["Visualization"]["plot_hist_solution"],
    config["Visualization"]["create_plots"],
)

hydragnn.utils.save_model(model, log_name)
hydragnn.utils.print_timers(verbosity)
##################################################################################################################
