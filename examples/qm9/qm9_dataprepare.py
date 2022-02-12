import os, json

import torch
import torch_geometric
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn


graph_feature_names = [
    "mu",
    "alpha",
    "HOMO",
    "LUMO",
    "del-epi",
    "R2",
    "ZPVE",
    "U0",
    "U",
    "H",
    "G",
    "cv",
    "U0atom",
    "Uatom",
    "Hatom",
    "Gatom",
    "A",
    "B",
    "C",
]
graph_feature_units = [
    "D",
    "a_0^3",
    "eV",
    "eV",
    "eV",
    "a_0^2",
    "eV",
    "eV",
    "eV",
    "eV",
    "eV",
    "cal/(molK)",
    "eV",
    "eV",
    "eV",
    "eV",
    "GHz",
    "GHz",
    "GHz",
]
node_attribute_names = [
    "atomH",
    "atomC",
    "atomN",
    "atomO",
    "atomF",
    "atomicnumber",
    "IsAromatic",
    "HSP",
    "HSP2",
    "HSP3",
    "Hprop",
    "chargedensity",
]
outlierlist = []
with open("dataset/qm9/outlierlist_ABC.txt") as f:
    for line in f:  # read rest of lines
        for x in line.split():
            outlierlist.append(int(x))
print(outlierlist)


def qm9_pre_filter_outlier(data):
    return data.idx not in outlierlist


dataset = torch_geometric.datasets.QM9_custom(
    root="dataset/qm9/total", pre_filter=qm9_pre_filter_outlier
)

############################################################
num_node_features = dataset.num_node_features
num_graph_features = dataset.num_classes

minmax_graph_feature = np.full((2, num_graph_features), np.inf)
minmax_node_feature = np.full((2, num_node_features), np.inf)

minmax_graph_feature_pernode = np.full((2, num_graph_features), np.inf)

for ifeat in range(num_graph_features):
    ymin, ymax = dataset.minmax_y(ifeat)
    minmax_graph_feature[0, ifeat] = ymin
    minmax_graph_feature[1, ifeat] = ymax

for ifeat in range(num_graph_features):
    ymin, ymax = dataset.minmax_y_pernode(ifeat)
    print(ifeat, ymin, ymax)
    minmax_graph_feature_pernode[0, ifeat] = ymin
    minmax_graph_feature_pernode[1, ifeat] = ymax

for ifeat in range(num_node_features):
    xmin, xmax = dataset.minmax_x(ifeat)
    minmax_node_feature[0, ifeat] = xmin
    minmax_node_feature[1, ifeat] = xmax
############################################################
print("dataset.data.idx", dataset.data.idx)
perc_train = 0.7
perc_val = (1 - perc_train) / 2
ntotal = dataset.data.idx.size(0)
ntrain = int(ntotal * perc_train)
nval = int(ntotal * perc_val)
ntest = ntotal - ntrain - nval
print(ntotal, ntrain, nval, ntest)
randomlist = torch.randperm(ntotal)

idx_train_list = dataset.data.idx[randomlist[:ntrain]]
idx_val_list = dataset.data.idx[randomlist[ntrain : ntrain + nval]]
idx_test_list = dataset.data.idx[randomlist[ntrain + nval :]]

print(idx_train_list, idx_val_list, idx_test_list)
############################################################
# print("trainset.data.idx",trainset.data.idx)
# print("valset.data.idx",valset.data.idx)
# print("testset.data.idx",testset.data.idx)

serial_data_name = "qm9_minmax.pkl"
with open(os.path.join("dataset/qm9", serial_data_name), "wb") as f:
    pickle.dump(minmax_node_feature, f)
    pickle.dump(minmax_graph_feature, f)
    pickle.dump(minmax_graph_feature_pernode, f)
    pickle.dump(node_attribute_names, f)
    pickle.dump(graph_feature_names, f)
    pickle.dump(graph_feature_units, f)

serial_data_name = "qm9_train_test_val_idx_lists.pkl"
with open(os.path.join("dataset/qm9", serial_data_name), "wb") as f:
    pickle.dump(idx_train_list, f)
    pickle.dump(idx_val_list, f)
    pickle.dump(idx_test_list, f)
############################################################
############################################################
############################################################
# Update each sample prior to loading.
def qm9_pre_transform(data):
    # remove constant x attributes
    data.x = (data.x - minmax_node_feature[0, :]) / (
        minmax_node_feature[1, :] - minmax_node_feature[0, :]
    )
    for ifeat, unit in enumerate(graph_feature_units):
        if unit == "eV":
            print("energy_related: ", graph_feature_names[ifeat])
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
        data.to(device)
        datasetlist.append(data)
del train, val, test
##################################################################################################################
##################################################################################################################
for ifeat in range(19):
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
        "dataset/qm9/" + "/qm9_train_val_test_" + graph_feature_names[ifeat] + ".png"
    )
    plt.close()

for ifeat in range(12):
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
        "dataset/qm9/" + "/qm9_train_val_test_" + node_attribute_names[ifeat] + ".png"
    )
    plt.close()
