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
import pickle
from scipy.interpolate import make_interp_spline, BSpline

plt.rcParams.update({"font.size": 16})
#########################################################

CaseDir = os.path.join(
    os.path.dirname(__file__),
    "../../logs/MultitaskingPaper_sequential/summit_lsms_hydra",
)
caseslabel = [
    "energy",
    "charge",
    "magnetic",
    "energy_charge",
    "energy_magnetic",
    "charge_magnetic",
    "energy_charge_magnetic",
]
line_labels = ["H", "C", "M", "HC", "HM", "CM", "HCM"]
variable_case_dic = {
    "energy": [[0, 0], [3, 0], [4, 0], [6, 0]],
    "charge": [[1, 0], [3, 1], [5, 0], [6, 1]],
    "magnetic": [[2, 0], [4, 1], [5, 1], [6, 2]],
}

######################################################################
linelib = ["-", "--", "-.", ":"]
colorlib = ["r", "g", "b", "m", "k"]
markerlib = ["o", "+", "s", "d", "^"]
linewidlib = [1.5, 1]
title_labels = ["Mixing enthalpy", "Charge transfer", "Magnetic moment"]
# line_labels = ["STL", "MTL-2", "MTL-3"]
######################################################################

fig, axs = plt.subplots(1, 3, figsize=(14, 5))
for ivar in range(3):
    var = caseslabel[ivar]

    icol = ivar
    ax = axs[icol]

    xmin = 1
    xmax = -1
    for isub in range(len(variable_case_dic[var])):
        icase = variable_case_dic[var][isub][0]
        ihead = variable_case_dic[var][isub][1]

        train_runs = []
        test_runs = []
        for irun in range(1, 11):
            case_name = "lsms_"
            if icase > 2:
                case_name += "multitask_"
            case_name += caseslabel[icase]
            model_name = case_name + "_" + str(irun)
            history_dir = CaseDir + "/logs/" + model_name + "/"
            if os.path.exists(history_dir + "/history_loss.pckl"):
                with open(history_dir + "/history_loss.pckl", "rb") as f:
                    (
                        total_loss_train,
                        total_loss_val,
                        total_loss_test,
                        task_loss_train,
                        task_loss_val,
                        task_loss_test,
                        task_weights,
                        task_names,
                    ) = pickle.load(f)
            train_history = [loss_[ihead] for loss_ in task_loss_train]
            test_history = [loss_[ihead] for loss_ in task_loss_test]

            train_runs.append(train_history)
            test_runs.append(test_history)
        ######################################################################
        train_runs = np.stack(train_runs, axis=0).squeeze()
        train_runs_mean = np.mean(train_runs, axis=0)
        train_runs_std = np.std(train_runs, axis=0)

        test_runs = np.stack(test_runs, axis=0)
        test_runs_mean = np.mean(test_runs, axis=0)
        test_runs_std = np.std(test_runs, axis=0)

        ic = icase // 3
        il = isub

        ax.plot(
            range(200),
            train_runs_mean,
            linelib[il],
            color=colorlib[ic],
            linewidth=2.0,
            label=line_labels[icase],
        )
        ax.fill_between(
            range(200),
            train_runs_mean - train_runs_std,
            train_runs_mean + train_runs_std,
            facecolor=colorlib[ic],
            interpolate=True,
            alpha=0.4,
        )
        ################################################################################################################

    if icol == 0:
        ax.set_ylabel("Train Loss", rotation=90, fontsize=18)
    ax.set_xlabel(r"epochs", fontsize=18)

    ax.title.set_text(title_labels[icol])

for icol in range(3):
    axs[icol].set_yscale("log")
    axs[icol].legend(fontsize=16, loc="upper right")
#    axs[icol].set_ylim(-5, 65)
#    if icol > 0:
#        axs[icol].set_yticklabels([])

fig.subplots_adjust(
    left=0.1, bottom=None, right=0.99, top=0.85, wspace=0.16, hspace=0.15
)
filenamepng = CaseDir + "/../Losshistory_SLT_MLT_ensemble.png"
plt.savefig(filenamepng, bbox_inches="tight")
filenamepng = CaseDir + "/../Losshistory_SLT_MLT_ensemble.pdf"
plt.savefig(filenamepng, bbox_inches="tight")
plt.close()
