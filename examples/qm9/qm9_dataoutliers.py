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


dataset = torch_geometric.datasets.QM9_custom(root="dataset/qm9")

############################################################
outlierlist = []
for ifeat in range(len(graph_feature_names) - 3, len(graph_feature_names)):
    print(graph_feature_names[ifeat])
    yfeat = dataset.get_y(ifeat).squeeze().detach().numpy()
    if graph_feature_names[ifeat] == "A":
        for item in np.argwhere(yfeat > 100).tolist():
            for itemitem in item:
                outlierlist.append(dataset[itemitem].idx.item())
        print(outlierlist)
    elif graph_feature_names[ifeat] == "B":
        for item in np.argwhere(yfeat > 10).tolist():
            for itemitem in item:
                outlierlist.append(dataset[itemitem].idx.item())
        print(outlierlist)
    else:
        for item in np.argwhere(yfeat > 10).tolist():
            for itemitem in item:
                outlierlist.append(dataset[itemitem].idx.item())
        print(outlierlist)
outlierlist = list(set(outlierlist))
with open("dataset/qm9/outlierlist_ABC.txt", "w") as texfile:
    texfile.write("\n".join(str(item) for item in outlierlist))
############################################################
num_node_features = dataset.num_node_features
num_graph_features = dataset.num_classes

minmax_graph_feature = np.full((2, num_graph_features), np.inf)
minmax_node_feature = np.full((2, num_node_features), np.inf)

minmax_graph_feature_pernode = np.full((2, num_graph_features), np.inf)

for ifeat in range(num_graph_features):
    ymin, ymax = dataset.minmax_y(ifeat)
    print(ifeat, ymin, ymax)
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
"""
for ifeat in range(len(graph_feature_names)):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    ) 
    yfeat = dataset.get_y(ifeat).squeeze().detach().numpy()
    print(yfeat.shape, yfeat)
    ax = axs[0]
    ax.scatter(range(yfeat.shape[0]), yfeat, edgecolor="b",facecolor="none")
    ax.scatter(outlierlist, [yfeat[item] for item in outlierlist], edgecolor="r",facecolor="none")
    ax.set_title("len=" + str(len(dataset))+", min={:.2f}".format(minmax_graph_feature[0,ifeat])+", max={:.2f}".format(minmax_graph_feature[1,ifeat])+", "+graph_feature_units[ifeat])
    ax = axs[1]
    hist1d,bine,_=ax.hist(yfeat,bins=100)
    binc = 0.5*(bine[0:-1]+bine[1:])
    #for i in range(len(hist1d)):
    #    if hist1d[i]<10:
    #       ax.plot([binc[i],binc[i]],[0,hist1d[i]*1000],'r-')
    fig.savefig(
        "dataset/plots"
        + "/qm9_dataset_"
        + graph_feature_names[ifeat]
        + ".png"
    )
    plt.close()

for ifeat in range(len(node_attribute_names)):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    xfeat = dataset.get_x(ifeat).squeeze().detach().numpy()
    ax = axs[0]
    ax.scatter(range(xfeat.shape[0]),xfeat, edgecolor="b",facecolor="none")
    ax.scatter(outlierlist, [xfeat[item] for item in outlierlist], edgecolor="r",facecolor="none")
    ax.set_title("len=" + str(len(dataset))+", min={:.2f}".format(minmax_node_feature[0,ifeat])+", max={:.2f}".format(minmax_node_feature[1,ifeat]))
    ax = axs[1]
    hist1d,bine,_=ax.hist(xfeat,bins=100)
    binc = 0.5*(bine[0:-1]+bine[1:])
    #for i in range(len(hist1d)):
    #    if hist1d[i]<10:
    #       ax.plot([binc[i],binc[i]],[0,hist1d[i]*1000],'r-')
    fig.savefig(
        "dataset/plots"
        + "/qm9_dataset_x_"
        + node_attribute_names[ifeat]
        + ".png"
    )
    plt.close()
"""
############################################################
import sys


def qm9_pre_filter_outlier(data):
    if data.idx in outlierlist:
        print(data.idx, outlierlist, data.idx in outlierlist)
    # if data.idx==100:
    #   sys.exit()
    return data.idx not in outlierlist


dataset = torch_geometric.datasets.QM9_custom(
    root="dataset/qm9/total", pre_filter=qm9_pre_filter_outlier
)
for ifeat in range(num_graph_features):
    ymin, ymax = dataset.minmax_y(ifeat)
    print(ifeat, ymin, ymax)
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
for ifeat in range(len(graph_feature_names)):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    yfeat = dataset.get_y(ifeat).squeeze().detach().numpy()
    print(yfeat.shape, yfeat)
    ax = axs[0]
    ax.scatter(range(yfeat.shape[0]), yfeat, edgecolor="b", facecolor="none")
    ax.scatter(
        outlierlist,
        [yfeat[item] for item in outlierlist],
        edgecolor="r",
        facecolor="none",
    )
    ax.set_title(
        "len="
        + str(len(dataset))
        + ", min={:.2f}".format(minmax_graph_feature[0, ifeat])
        + ", max={:.2f}".format(minmax_graph_feature[1, ifeat])
        + ", "
        + graph_feature_units[ifeat]
    )
    ax = axs[1]
    hist1d, bine, _ = ax.hist(yfeat, bins=100)
    binc = 0.5 * (bine[0:-1] + bine[1:])
    for i in range(len(hist1d)):
        if hist1d[i] < 10:
            ax.plot([binc[i], binc[i]], [0, hist1d[i] * 1000], "g-")
    fig.savefig(
        "dataset/plots" + "/cutoff_qm9_dataset_" + graph_feature_names[ifeat] + ".png"
    )
    plt.close()

for ifeat in range(len(node_attribute_names)):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    xfeat = dataset.get_x(ifeat).squeeze().detach().numpy()
    ax = axs[0]
    ax.scatter(range(xfeat.shape[0]), xfeat, edgecolor="b", facecolor="none")
    ax.scatter(
        outlierlist,
        [xfeat[item] for item in outlierlist],
        edgecolor="r",
        facecolor="none",
    )
    ax.set_title(
        "len="
        + str(len(dataset))
        + ", min={:.2f}".format(minmax_node_feature[0, ifeat])
        + ", max={:.2f}".format(minmax_node_feature[1, ifeat])
    )
    ax = axs[1]
    hist1d, bine, _ = ax.hist(xfeat, bins=100)
    binc = 0.5 * (bine[0:-1] + bine[1:])
    for i in range(len(hist1d)):
        if hist1d[i] < 10:
            ax.plot([binc[i], binc[i]], [0, hist1d[i] * 1000], "g-")
    fig.savefig(
        "dataset/plots"
        + "/cutoff_qm9_dataset_x_"
        + node_attribute_names[ifeat]
        + ".png"
    )
    plt.close()
