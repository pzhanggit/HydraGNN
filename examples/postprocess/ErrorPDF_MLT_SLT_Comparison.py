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
variable_case_dic = {
    "energy": [[0, 0], [3, 0], [4, 0], [6, 0]],
    "charge": [[1, 0], [3, 1], [5, 0], [6, 1]],
    "magnetic": [[2, 0], [4, 1], [5, 1], [6, 2]],
}

for irun in range(7, 13):
    error_list = [None] * len(caseslabel)
    pdf_list = [None] * len(caseslabel)
    xcen_list = [None] * len(caseslabel)
    if not os.path.exists(CaseDir + "/PDFofError_SLT_MLT_" + str(irun) + ".pkl"):
        for icase in range(0, len(caseslabel)):
            config_file = CaseDir + "/inputs/"
            case_name = "lsms_"
            if icase > 2:
                case_name += "multitask_"
            case_name += caseslabel[icase]
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
            xcen_heads = []
            pdfs_heads = []
            for ihead in range(model.num_heads):
                pdf1d, bin_edges = np.histogram(
                    np.array(predicted_values[ihead]) - np.array(true_values[ihead]),
                    bins=200,
                    density=True,
                )

                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                xcen_heads.append(bin_centers)
                pdfs_heads.append(pdf1d)

            error_list[icase] = error_rmse_task
            xcen_list[icase] = xcen_heads
            pdf_list[icase] = pdfs_heads
        ######################################################################
        with open(CaseDir + "/../PDFofError_SLT_MLT_" + str(irun) + ".pkl", "wb") as f:
            pickle.dump(error_list, f)
            pickle.dump(xcen_list, f)
            pickle.dump(pdf_list, f)
    else:
        with open(CaseDir + "/../PDFofError_SLT_MLT_" + str(irun) + ".pkl", "rb") as f:
            error_list = pickle.load(f)
            xcen_list = pickle.load(f)
            pdf_list = pickle.load(f)
    ######################################################################
    linelib = ["-", "--", "-.", ":"]
    colorlib = ["r", "g", "b", "m", "k"]
    markerlib = ["o", "+", "s", "d", "^"]
    linewidlib = [1.5, 1]
    title_labels = ["Mixing enthalpy", "Charge transfer", "Magnetic moment"]
    line_labels = ["STL", "MTL-2", "MTL-3"]
    ######################################################################
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    for ivar in range(3):
        var = caseslabel[ivar]

        icol = ivar
        ax = axs[icol]
        for isub in range(len(variable_case_dic[var])):
            icase = variable_case_dic[var][isub][0]
            ihead = variable_case_dic[var][isub][1]

            xcen = xcen_list[icase][ihead]
            pdf1d = pdf_list[icase][ihead]
            err_rmse = error_list[icase][ihead]

            ic = icase // 3
            il = isub
            print(icol, icase, il, ic, err_rmse)
            (II,) = np.where(pdf1d > 1.0)
            # xnew = np.linspace(xcen[II].min(), xcen[II].max(), 50)
            # spl = make_interp_spline(xcen[II], pdf1d[II], k=1)
            # ysmooth = spl(xnew)

            ax.plot(
                xcen[II],
                pdf1d[II],
                linelib[il],
                color=colorlib[ic],
                linewidth=2.0,
                label=line_labels[ic],
            )
            ax.plot([0, 0], [0, 55], "k--")
        if icol == 0:
            ax.set_ylabel("PDF", rotation=90, fontsize=18)
        ax.set_xlabel(r"$(y_{pred}-y_{true})/(y_{max}-y_{min})$", fontsize=18)

        ax.title.set_text(title_labels[icol])
    axs[0].legend(fontsize=16)
    fig.subplots_adjust(
        left=0.1, bottom=None, right=0.99, top=0.85, wspace=0.18, hspace=0.15
    )
    filenamepng = CaseDir + "/../PDFofError_SLT_MLT_" + str(irun) + ".png"
    plt.savefig(filenamepng, bbox_inches="tight")
    plt.close()
