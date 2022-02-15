import os, json

import torch
import torch_geometric

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn
import pickle
import matplotlib.pyplot as plt
import sys

serial_data_name = "qm9_minmax.pkl"
with open(os.path.join("dataset/qm9", serial_data_name), "rb") as f:
    minmax_node_feature = torch.Tensor(pickle.load(f))
    minmax_graph_feature = torch.Tensor(pickle.load(f))
    minmax_graph_feature_pernode = torch.Tensor(pickle.load(f))
    node_attribute_names = pickle.load(f)
    graph_feature_names = pickle.load(f)
    graph_feature_units = pickle.load(f)
assert len(graph_feature_names) == len(
    graph_feature_units
), "mismatched units for graph features"

node_feature_nonconstant_indices = (
    (minmax_node_feature[0, :] < minmax_node_feature[1, :])
    .nonzero(as_tuple=True)[0]
    .tolist()
)

node_attribute_names_custom = [
    node_attribute_names[item] for item in node_feature_nonconstant_indices
]

serial_data_name = "qm9_train_test_val_idx_lists.pkl"
with open(os.path.join("dataset/qm9", serial_data_name), "rb") as f:
    idx_train_list = pickle.load(f)
    idx_val_list = pickle.load(f)
    idx_test_list = pickle.load(f)


# Update each sample prior to loading.
def qm9_pre_transform(data):
    # remove constant x attributes
    data.x = (
        data.x[:, node_feature_nonconstant_indices]
        - minmax_node_feature[0, node_feature_nonconstant_indices]
    ) / (
        minmax_node_feature[1, node_feature_nonconstant_indices]
        - minmax_node_feature[0, node_feature_nonconstant_indices]
    )
    for ifeat, unit in enumerate(graph_feature_units):
        if unit == "eV":
            data.y[:, ifeat] = data.y[:, ifeat] / data.x.size(0)
            data.y[:, ifeat] = (
                data.y[:, ifeat] - minmax_graph_feature_pernode[0, ifeat]
            ) / (
                minmax_graph_feature_pernode[1, ifeat]
                - minmax_graph_feature_pernode[0, ifeat]
            )
        else:
            data.y[:, ifeat] = (data.y[:, ifeat] - minmax_graph_feature[0, ifeat]) / (
                minmax_graph_feature[1, ifeat] - minmax_graph_feature[0, ifeat]
            )

    data.y = data.y.squeeze()
    device = hydragnn.utils.get_device()
    return data.to(device)


def qm9_pre_filter_train(data):
    return data.idx in idx_train_list


def qm9_pre_filter_val(data):
    return data.idx in idx_val_list


def qm9_pre_filter_test(data):
    return data.idx in idx_test_list


# Set this path for output.
try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

inputfilesubstr = sys.argv[1]
# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(__file__), "qm9_" + inputfilesubstr + ".json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]
var_config["output_names"] = [
    graph_feature_names[item]
    if var_config["type"][ihead] == "graph"
    else node_attribute_names_custom[item]
    for ihead, item in enumerate(var_config["output_index"])
]
var_config["input_node_feature_names"] = [
    node_attribute_names_custom[item] for item in var_config["input_node_features"]
]
config["NeuralNetwork"]["Training"]["num_epoch"] = 300
# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()

# Use built-in torch_geometric dataset.
# Filter function above used to run quick example.
# NOTE: data is moved to the device in the pre-transform.
# NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
train = torch_geometric.datasets.QM9_custom(
    root="dataset/qm9/train",
    pre_transform=qm9_pre_transform,
    pre_filter=qm9_pre_filter_train,
)
val = torch_geometric.datasets.QM9_custom(
    root="dataset/qm9/val",
    pre_transform=qm9_pre_transform,
    pre_filter=qm9_pre_filter_val,
)
test = torch_geometric.datasets.QM9_custom(
    root="dataset/qm9/test",
    pre_transform=qm9_pre_transform,
    pre_filter=qm9_pre_filter_test,
)
##################################################################################################################
trainset = []
valset = []
testset = []
device = hydragnn.utils.get_device()
for dataset, datasetlist in zip([train, val, test], [trainset, valset, testset]):
    for data in dataset:
        hydragnn.preprocess.update_predicted_values(
            var_config["type"],
            var_config["output_index"],
            data,
        )
        hydragnn.preprocess.update_atom_features(
            var_config["input_node_features"], data
        )
        data.to(device)
        datasetlist.append(data)
del train, val, test
##################################################################################################################
(
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
) = hydragnn.preprocess.create_dataloaders(
    trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
)


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

# Run training with the given model and qm9 dataset.
log_name = "qm9_test_" + inputfilesubstr
writer = hydragnn.utils.get_summary_writer(log_name)
with open("./logs/" + log_name + "/config.json", "w") as f:
    json.dump(config, f)
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
)

save_model(model, log_name)
##################################################################################################################
for ifeat in range(len(var_config["output_index"])):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    ax = axs[0]
    ax.scatter(
        [trainset[i].cpu().idx for i in range(len(trainset))],
        [trainset[i].cpu().y[ifeat] for i in range(len(trainset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("train, " + str(len(trainset)))
    ax = axs[1]
    ax.scatter(
        [valset[i].cpu().idx for i in range(len(valset))],
        [valset[i].cpu().y[ifeat] for i in range(len(valset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("validate, " + str(len(valset)))
    ax = axs[2]
    ax.scatter(
        [testset[i].cpu().idx for i in range(len(testset))],
        [testset[i].cpu().y[ifeat] for i in range(len(testset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("test, " + str(len(testset)))
    fig.savefig(
        "./logs/"
        + log_name
        + "/qm9_train_val_test_"
        + var_config["output_names"][ifeat]
        + ".png"
    )
    plt.close()

for ifeat in range(len(var_config["input_node_features"])):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    ax = axs[0]
    ax.plot(
        [
            item
            for i in range(len(trainset))
            for item in trainset[i].x[:, ifeat].tolist()
        ],
        "bo",
    )
    ax.set_title("train, " + str(len(trainset)))
    ax = axs[1]
    ax.plot(
        [item for i in range(len(valset)) for item in valset[i].x[:, ifeat].tolist()],
        "bo",
    )
    ax.set_title("validate, " + str(len(valset)))
    ax = axs[2]
    ax.plot(
        [item for i in range(len(testset)) for item in testset[i].x[:, ifeat].tolist()],
        "bo",
    )
    ax.set_title("test, " + str(len(testset)))
    fig.savefig(
        "./logs/"
        + log_name
        + "/qm9_train_val_test_"
        + var_config["input_node_feature_names"][ifeat]
        + ".png"
    )
    plt.close()
