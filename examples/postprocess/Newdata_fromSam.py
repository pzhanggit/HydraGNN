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

import json, os
import matplotlib.pyplot as plt

from hydragnn.preprocess.load_data import dataset_loading_and_splitting
from hydragnn.utils.distributed import setup_ddp
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.config_utils import (
    update_config,
    get_log_name_config,
)
from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import test
import numpy as np
from scipy.interpolate import griddata
import torch


plt.rcParams.update({"font.size": 16})
#########################################################

CaseDir = os.path.join(
    os.path.dirname(__file__),
    "../../logs/MultitaskingPaper_sequential/summit_lsms_hydra",
)
caseslabel = ["energy"]
######################################################################

def plot_results(composition_list,formation_enthalpy_list):
    #################################################
    comp_unq = sorted(list(set(composition_list)))
    com_freq = [composition_list.count(comp_) for comp_ in comp_unq]
    # hist1d, xbins_edge = np.histogram(composition_list, bins=nbin)
    # xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])

    fig = plt.figure(figsize=(7, 6))
    # plt.plot(xbin_cen, hist1d)
    plt.plot(comp_unq, com_freq, "ro-")
    plt.xlabel("Fe concentration")
    plt.ylabel("Hist1d")
    plt.title("FePt: " + str(len(composition_list)))
    fig.subplots_adjust(
        left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.1, hspace=0.06
    )
    plt.savefig(CaseDir + "/../histogram_Fe_concentration.png", dpi=400)
    ###############################
    nbin = 20
    xbins_edge = [comp_unq_ - 1.0 / 32.0 for comp_unq_ in comp_unq]
    xbins_edge.append(1 + 1.0 / 32.0)
    print(xbins_edge)
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=composition_list, y=formation_enthalpy_list, bins=[xbins_edge, nbin]
    )
    print(xbins_edge)

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
        (composition_list, formation_enthalpy_list),
        method="linear",
        fill_value=0,
    )  # np.nan)
    #################################################
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(
        composition_list, formation_enthalpy_list, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("Fe concentration")
    plt.ylabel("Mixing enthalpy (Rydberg)")
    plt.title("FePt")
    fig.subplots_adjust(
        left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.1, hspace=0.06
    )
    plt.savefig(CaseDir + "/../formation_enthalpy_colored.png", dpi=400)
    plt.savefig(CaseDir + "/../formation_enthalpy_colored.pdf")
##################################################################################################
linelib = ["-", "--", "-.", ":"]
colorlib = ["r", "g", "b", "m", "k"]
markerlib = ["o", "+", "s", "d", "^"]
linewidlib = [1.5, 1]
title_labels = ["Mixing enthalpy", "Charge transfer", "Magnetic moment"]
line_labels = ["HCM"]
unit_labels = ["(Rydberg)", "(electron charge)", "(magneton)"]
######################################################################
icase=0

config_file = CaseDir + "/inputs/"
case_name = "lsms_" + caseslabel[icase]
with open(config_file + case_name + ".json", "r") as f:
    config = json.load(f)

os.environ["SERIALIZED_DATA_PATH"] = CaseDir

world_size, world_rank = setup_ddp()

train_loader, val_loader, test_loader = dataset_loading_and_splitting(
    config=config
)

config = update_config(config, train_loader, val_loader, test_loader)

xminmax = config["NeuralNetwork"]["Variables_of_interest"]["x_minmax"][0]
yminmax = config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"][0]

log_name = get_log_name_config(config)
composition_list = []
formation_enthalpy_list = []
for loader in [train_loader, val_loader, test_loader]:
    for data in loader.dataset:
        proton_num = data.x[:,0]*(xminmax[1]-xminmax[0]) +xminmax[0]
        H = data.y[0,0].item()*(yminmax[1]-yminmax[0]) +yminmax[0]
        n_fe = torch.sum(torch.abs(proton_num-26)<1e-2).item()
        formation_enthalpy_list.append(H*32) ###it is scaled by num_nodes in new runs
        composition_list.append(n_fe/len(proton_num))



plot_results(composition_list,formation_enthalpy_list)





