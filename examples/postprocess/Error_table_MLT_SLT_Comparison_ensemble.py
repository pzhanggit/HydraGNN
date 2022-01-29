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
head_dict = {"H": 0, "C": 1, "M": 2}
######################################################################

xmin = 1
xmax = -1
for icase in [6, 3, 4, 5, 0, 1, 2]:
    ihead = -1
    tab_string = ["-", "-", "-"]
    for head in line_labels[icase]:
        ihead += 1
        ivar = head_dict[head]
        rmse_runs = []
        for irun in range(1, 11):
            if os.path.exists(CaseDir + "/../PDFofError_SLT_MLT_" + str(irun) + ".pkl"):
                with open(
                    CaseDir + "/../PDFofError_SLT_MLT_" + str(irun) + ".pkl", "rb"
                ) as f:
                    error_list = pickle.load(f)
                    xcen_list = pickle.load(f)
                    pdf_list = pickle.load(f)

            err_rmse = error_list[icase][ihead]
            rmse_runs.append(err_rmse)
        # print (ihead, icase, ivar, head)
        if head == "H":
            print(rmse_runs)
        rmse_mean = np.mean(np.asarray(rmse_runs))
        rmse_std = np.std(np.asarray(rmse_runs))
        tab_string[ivar] = "${:.2e}".format(rmse_mean) + "\pm{:.2e}$".format(rmse_std)
        # print(title_labels[ivar], line_labels[icase], rmse_mean, rmse_std)
        # if icase>3:
        #    print("&{:.2e}".format(rmse_mean)+"$\pm${:.2e}".format(rmse_std))
        # else:
        #    print("&{:.2e}".format(rmse_mean)+"$\pm${:.2e}".format(rmse_std))

# if icase > 2:
#     print("MTL, " + line_labels[icase]+"&"+"&".join(tab_string)+r"\\")
# else:
#     print("STL, " + line_labels[icase]+"&"+"&".join(tab_string)+r"\\")
