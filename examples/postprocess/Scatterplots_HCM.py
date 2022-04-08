##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os
import re
import numpy as np
import pickle
import pathlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

import torch
from torch_geometric.data import Data
from torch import tensor
from os import walk
from scipy.interpolate import griddata


plt.rcParams.update({"font.size": 16})

def load_raw_data(path,tag=""):
    """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    After that the serialized data is stored to the serialized_dataset directory.
    """
    #pkfile = os.path.join(path, "../FePt.pk")
    pkfile = os.path.join(path, "../FePt"+tag+".pk")
    if os.path.isfile(pkfile):
        with open(pkfile, "rb") as f:
            dataset=pickle.load(f)
            return dataset
    file_list = []
    for (dirpath, dirnames, filenames) in walk(path):
        file_list.extend(filenames)
        break

    dataset = []

    for filename in file_list:
        # Using readlines()
        filepath = path + '/' + filename

        data_object = transform_LSMS_input_to_data_object_base(filepath)

        dataset.append(data_object)

    with open(pkfile, "wb") as f:
        pickle.dump(dataset, f)

    return dataset

def transform_LSMS_input_to_data_object_base(filepath):
    """Transforms lines of strings read from the raw data LSMS file to Data object and returns it.
    Parameters
    ----------
    lines:
      content of data file with all the graph information
    Returns
    ----------
    Data
        Data object representing structure of a graph sample.
    """

    data_object = Data()

    f = open(filepath, "r", encoding="utf-8")

    lines = f.readlines()
    graph_feat = lines[0].split(None, 2)
    g_feature = []
    # collect graph features
    g_feature.append(float(graph_feat[0]))
    data_object.y = tensor(g_feature)

    node_feature_matrix = []
    node_position_matrix = []
    for line in lines[1:]:
        node_feat = line.split(None, 11)

        x_pos = float(node_feat[2].strip())
        y_pos = float(node_feat[3].strip())
        z_pos = float(node_feat[4].strip())
        node_position_matrix.append([x_pos, y_pos, z_pos])

        node_feature = []

        node_feature.append(float(node_feat[0]))
        node_feature.append(float(node_feat[5]))
        node_feature.append(float(node_feat[6]))
        node_feature_matrix.append(node_feature)

    f.close()

    data_object.pos = tensor(node_position_matrix)
    data_object.x = tensor(node_feature_matrix)
    data_object.total_magnetic_moment = torch.sum(data_object.x[:,2])
    return data_object

def getcolordensity(xdata, ydata):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=xdata, y=ydata, bins=[nbin, nbin]
    )
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm
if __name__ == '__main__':
    #path = '../../../dataset/FePt_enthalpy'
    path = '../../../dataset/FePt_gibbs_energy'
    tag="_sam"
    #tag=""
    dataset = load_raw_data(path,tag=tag)
    #################################################
    fig = plt.figure(figsize=(7, 6))
    xdata = [data.total_magnetic_moment.item() for data in dataset]
    ydata = [data.y.item() for data in dataset]
    hist2d_norm=getcolordensity(xdata, ydata)

    plt.scatter(
        xdata, ydata, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("Total magnetic moment (magneton)")
    plt.ylabel("Mixing enthalpy (Rydberg)")
    plt.title("FePt")
    fig.subplots_adjust(
        left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.1, hspace=0.06
    )
    plt.savefig( "./totalM_H"+tag+".png", dpi=400)

    fig = plt.figure(figsize=(7, 6))
    xdata = [sum(data.x[:,0]==26.0).item() for data in dataset]
    ydata = [data.total_magnetic_moment.item() for data in dataset]
    hist2d_norm = getcolordensity(xdata, ydata)

    plt.scatter(
        xdata, ydata, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("Fe concentration")
    plt.ylabel("Total magnetic moment (magneton)")
    plt.title("FePt")
    fig.subplots_adjust(
        left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.1, hspace=0.06
    )
    plt.savefig("./Feconcentration_totalM"+tag+".png", dpi=400)

    fig, axs = plt.subplots(1, 3,figsize=(15, 5))
    for isub in range(3):
        ax = axs[isub]
       # num_of_protons = data_object.x[:, 0]
       # charge_density = data_object.x[:, 1]
       # charge_density -= num_of_protons
        if isub==0:
            xdata = [data.x[inode, 2].item() for data in dataset for inode in range(data.x.shape[0])] #magnetic moment
            ydata = [data.y.item() for data in dataset for inode in range(data.x.shape[0])] #repeat for global quantities also
            xtag = "Magnetic moment (magneton)"
            ytag = "Mixing enthalpy (Rydberg)"
        elif isub==1:
            xdata = [data.x[inode, 1].item()-data.x[inode, 0].item() for data in dataset for inode in range(data.x.shape[0])] #charge density
            ydata = [data.y.item() for data in dataset for inode in range(data.x.shape[0])]  # repeat for global quantities also
            xtag = "Charge transfer (electron charge)"
            ytag = "Mixing enthalpy (Rydberg)"
        else:
            xdata = [data.x[inode, 2].item() for data in dataset for inode in range(data.x.shape[0])]
            ydata = [data.x[inode, 1].item()-data.x[inode, 0].item()  for data in dataset for inode in range(data.x.shape[0])]
            xtag = "Magnetic moment (magneton)"
            ytag = "Charge transfer (electron charge)"

        hist2d_norm = getcolordensity(xdata, ydata)
        im=ax.scatter(
            xdata, ydata, s=8, c=hist2d_norm, vmin=0, vmax=1
        )
        ax.set_xlabel(xtag)
        ax.set_ylabel(ytag)
    cbar_ax = fig.add_axes([0.2, 0.06, 0.75, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    fig.subplots_adjust(
        left=0.08, bottom=0.25, right=0.99, top=0.95, wspace=0.25, hspace=0.06
    )
    plt.savefig("./MH_CH_MC"+tag+".png", dpi=400)




