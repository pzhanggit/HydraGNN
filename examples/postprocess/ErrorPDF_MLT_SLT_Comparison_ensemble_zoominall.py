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
colorlib = ["r", "g", "tab:olive", "b", "k"]
markerlib = ["o", "+", "s", "d", "^"]
linewidlib = [1.5, 1]
title_labels = ["Mixing enthalpy", "Charge transfer", "Magnetic moment"]
# line_labels = ["STL", "MTL-2", "MTL-3"]
######################################################################

fig, axs = plt.subplots(1, 3, figsize=(14, 5))
axins = []
for icol in range(3):
    if icol == 1:
        axins.append(axs[icol].inset_axes([0.02, 0.55, 0.3, 0.4]))
    else:
        axins.append(axs[icol].inset_axes([0.08, 0.55, 0.25, 0.4]))
for ivar in range(3):
    var = caseslabel[ivar]

    icol = ivar
    ax = axs[icol]

    xmin = 1
    xmax = -1
    for isub in range(len(variable_case_dic[var])):
        icase = variable_case_dic[var][isub][0]
        ihead = variable_case_dic[var][isub][1]

        rmse_runs = []
        xcen_runs = []
        pdf_runs = []
        for irun in range(1, 13):
            error_list = [None] * len(caseslabel)
            pdf_list = [None] * len(caseslabel)
            xcen_list = [None] * len(caseslabel)
            if os.path.exists(CaseDir + "/../PDFofError_SLT_MLT_" + str(irun) + ".pkl"):
                with open(
                    CaseDir + "/../PDFofError_SLT_MLT_" + str(irun) + ".pkl", "rb"
                ) as f:
                    error_list = pickle.load(f)
                    xcen_list = pickle.load(f)
                    pdf_list = pickle.load(f)

            xcen = xcen_list[icase][ihead]
            pdf1d = pdf_list[icase][ihead]
            err_rmse = error_list[icase][ihead]

            xcen_runs.append(xcen)
            pdf_runs.append(pdf1d)
            rmse_runs.append(err_rmse)

            (II,) = np.where(pdf1d > 0.5)
            xmin = np.minimum(xmin, xcen[II[0]])
            xmax = np.maximum(xmax, xcen[II[-1]])

        ######################################################################
        xnew = np.linspace(xmin, xmax, 100)
        pdf_runs_uni = []
        for irun in range(1, 11):
            pdf_runs_uni.append(
                np.interp(xnew, xcen_runs[irun - 1], pdf_runs[irun - 1])
            )
        pdf_runs_array = np.stack(pdf_runs_uni, axis=0)
        pdf_runs_mean = np.mean(pdf_runs_uni, axis=0)
        pdf_runs_std = np.std(pdf_runs_uni, axis=0)

        rmse_mean = np.mean(np.asarray(rmse_runs))
        rmse_std = np.std(np.asarray(rmse_runs))
        print(title_labels[icol], line_labels[icase], rmse_mean, rmse_std)

        ic = isub  # icase // 3
        il = isub

        ax.plot(
            xnew,
            pdf_runs_mean,
            linelib[il],
            color=colorlib[ic],
            linewidth=2.0,
            label=line_labels[icase],
            zorder=50,
        )
        ax.fill_between(
            xnew,
            pdf_runs_mean - pdf_runs_std,
            pdf_runs_mean + pdf_runs_std,
            facecolor=colorlib[ic],
            interpolate=True,
            alpha=0.4,
            zorder=10,
        )
        ax.plot([0, 0], [-5, 125], ":", color="grey", linewidth=1.0, zorder=0)
        ################################################################################################################
        axins[icol].plot(
            xnew,
            pdf_runs_mean,
            linelib[il],
            color=colorlib[ic],
            linewidth=2.0,
            label=line_labels[icase],
            zorder=50,
        )
        axins[icol].fill_between(
            xnew,
            pdf_runs_mean - pdf_runs_std,
            pdf_runs_mean + pdf_runs_std,
            facecolor=colorlib[ic],
            interpolate=True,
            alpha=0.4,
            zorder=10,
        )
        axins[icol].plot([0, 0], [0, 125], ":", color="grey", linewidth=1.0, zorder=0)
        axins[icol].set_xlim(-0.005, 0.005)  # apply the x-limits
        if icol == 0:
            axins[icol].set_xlim(-0.008, 0.008)  # apply the x-limits
            axins[icol].set_ylim(45, 120)  # apply the y-limits
        elif icol == 1:
            axins[icol].set_ylim(75, 120)  # apply the y-limits
        else:
            axins[icol].set_ylim(60, 95)  # apply the y-limits

        axins[icol].set_yticklabels([])
        axins[icol].set_xticklabels([])
        ax.indicate_inset_zoom(axins[icol])
        ################################################################################################################

    if icol == 0:
        ax.set_ylabel("PDF", rotation=90, fontsize=18)
    ax.set_xlabel(r"$(y_{pred}-y_{true})/(y_{max}-y_{min})$", fontsize=18)

    ax.title.set_text(title_labels[icol])

for icol in range(3):
    axs[icol].legend(fontsize=16, loc="upper right")
    axs[icol].set_ylim(-5, 125)
    if icol > 0:
        axs[icol].set_yticklabels([])

fig.subplots_adjust(
    left=0.1, bottom=None, right=0.99, top=0.85, wspace=0.05, hspace=0.15
)
filenamepng = CaseDir + "/../PDFofError_SLT_MLT_ensemble_zoominall.png"
plt.savefig(filenamepng, bbox_inches="tight")
filenamepng = CaseDir + "/../PDFofError_SLT_MLT_ensemble_zoominall.pdf"
plt.savefig(filenamepng, bbox_inches="tight")
plt.close()
