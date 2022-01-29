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

plt.rcParams.update({"font.size": 16})
#########################################################

CaseDir = os.path.join(
    os.path.dirname(__file__),
    "../../logs/MultitaskingPaper_sequential/summit_lsms_hydra",
)
caseslabel = ["energy_charge_magnetic"]
######################################################################
linelib = ["-", "--", "-.", ":"]
colorlib = ["r", "g", "b", "m", "k"]
markerlib = ["o", "+", "s", "d", "^"]
linewidlib = [1.5, 1]
title_labels = ["Mixing enthalpy", "Charge transfer", "Magnetic moment"]
line_labels = ["HCM"]
unit_labels = ["(Rydberg)", "(electron charge)", "(magneton)"]
######################################################################
for irun in range(1, 13):
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    for icase in range(1):
        config_file = CaseDir + "/inputs/"
        case_name = "lsms_multitask_" + caseslabel[icase]
        with open(config_file + case_name + ".json", "r") as f:
            config = json.load(f)

        os.environ["SERIALIZED_DATA_PATH"] = CaseDir

        world_size, world_rank = setup_ddp()

        train_loader, val_loader, test_loader = dataset_loading_and_splitting(
            config=config
        )

        config = update_config(config, train_loader, val_loader, test_loader)

        model = create_model_config(
            config=config["NeuralNetwork"]["Architecture"],
            verbosity=config["Verbosity"]["level"],
        )

        log_name = get_log_name_config(config)
        model_name = case_name + "_" + str(irun)

        load_existing_model(
            model, model_name, path=CaseDir + "/logs/" + model_name + "/"
        )

        (
            error,
            error_rmse_task,
            true_values,
            predicted_values,
        ) = test(test_loader, model, config["Verbosity"]["level"])

        for ihead in range(model.num_heads):

            ax = axs[ihead]

            head_min = config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"][
                ihead
            ][0]
            head_max = config["NeuralNetwork"]["Variables_of_interest"]["y_minmax"][
                ihead
            ][1]
            head_true = (
                np.asarray(true_values[ihead]) * (head_max - head_min) + head_min
            )
            head_pred = (
                np.asarray(predicted_values[ihead]) * (head_max - head_min) + head_min
            )

            ax.scatter(
                head_true,
                head_pred,
                s=7,
                linewidth=0.5,
                edgecolor="b",
                facecolor="none",
            )
            ax.plot([head_min, head_max], [head_min, head_max], "r--")
            # ax.set_ylabel(r"HydraGNN values "+unit_labels[icol], rotation=90, fontsize=18)
            # ax.set_xlabel(r"DFT values "+unit_labels[icol], fontsize=18)
            if ihead == 0:
                ax.set_ylabel("HydraGNN values", rotation=90, fontsize=18)
            ax.set_xlabel("DFT values", fontsize=18)

            ax.set_title(title_labels[ihead] + " " + unit_labels[ihead], fontsize=16)
    fig.subplots_adjust(
        left=0.1, bottom=None, right=0.99, top=0.85, wspace=0.1, hspace=0.15
    )
    filenamepng = (
        CaseDir + "/../Scatter_" + line_labels[icase] + "_" + str(irun) + ".png"
    )
    plt.savefig(filenamepng, bbox_inches="tight", dpi=400)
    filenamepng = (
        CaseDir + "/../Scatter_" + line_labels[icase] + "_" + str(irun) + ".pdf"
    )
    plt.savefig(filenamepng, bbox_inches="tight")
    plt.close()
