import os, json
import matplotlib.pyplot as plt
from ogb_utils import *

import logging
import sys
from tqdm import tqdm
import mpi4py

from mpi4py import MPI
from itertools import chain
import argparse
import time

from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.ogbdataset import AdiosOGB, OGBDataset
from hydragnn.utils.model import print_model

import numpy as np
import adios2 as ad2

import torch_geometric.data
import torch
import torch.distributed as dist

try:
    import gptl4py as gp
except ImportError:
    import hydragnn.utils.gptl4py_dummy as gp

import warnings

warnings.filterwarnings("error")


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


## Torch Dataset for CSCE CSV format
class OGBRawDatasetFactory:
    def __init__(self, datafile, var_config, sampling=1.0, seed=43, norm_yflag=False):
        self.var_config = var_config

        ## Read full data
        smiles_sets, values_sets = datasets_load(datafile, sampling=sampling, seed=seed)
        ymean = var_config["ymean"]
        ystd = var_config["ystd"]

        info([len(x) for x in values_sets])
        self.dataset_lists = list()
        for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
            if norm_yflag:
                valueset = (valueset - torch.tensor(ymean)) / torch.tensor(ystd)
                # print(valueset[:, 0].mean(), valueset[:, 0].std())
                # print(valueset[:, 1].mean(), valueset[:, 1].std())
                # print(valueset[:, 2].mean(), valueset[:, 2].std())
            self.dataset_lists.append((smileset, valueset))

    def get(self, label):
        ## Set only assigned label data
        labelnames = ["trainset", "valset", "testset"]
        index = labelnames.index(label)

        smileset, valueset = self.dataset_lists[index]
        return (smileset, valueset)


class OGBRawDataset(torch.utils.data.Dataset):
    def __init__(self, datasetfactory, label):
        self.smileset, self.valueset = datasetfactory.get(label)
        self.var_config = datasetfactory.var_config

    def __len__(self):
        return len(self.smileset)

    @gp.profile
    def __getitem__(self, idx):
        smilestr = self.smileset[idx]
        ytarget = self.valueset[idx]
        data = generate_graphdata(smilestr, ytarget, self.var_config)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfilesubstr", help="input file substr")
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios saving and no train",
    )
    parser.add_argument("--shmem", action="store_true", help="use shmem")
    parser.add_argument("--noadios", action="store_true", help="no adios dataset")
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    args = parser.parse_args()

    graph_feature_names = ["GAP"]
    dirpwd = os.path.dirname(__file__)
    datafile = os.path.join(dirpwd, "dataset/pcqm4m_gap.csv")
    trainset_statistics = os.path.join(dirpwd, "dataset/statistics.pkl")
    ##################################################################################################################
    inputfilesubstr = args.inputfilesubstr
    input_filename = os.path.join(dirpwd, "ogb_" + inputfilesubstr + ".json")
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["output_names"] = [
        graph_feature_names[item]
        for ihead, item in enumerate(var_config["output_index"])
    ]
    var_config["input_node_feature_names"] = node_attribute_names
    ymax_feature, ymin_feature, ymean_feature, ystd_feature = get_trainset_stat(
        trainset_statistics
    )
    var_config["ymean"] = ymean_feature.tolist()
    var_config["ystd"] = ystd_feature.tolist()
    ##################################################################################################################
    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "ogb_" + inputfilesubstr + "_eV_fullx"
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)
    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)

    if args.preonly:
        norm_yflag = False  # True
        smiles_sets, values_sets = datasets_load(
            datafile, sampling=args.sampling, seed=43
        )
        info([len(x) for x in values_sets])
        dataset_lists = [[] for dataset in values_sets]
        for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
            if norm_yflag:
                valueset = (
                    valueset - torch.tensor(var_config["ymean"])
                ) / torch.tensor(var_config["ystd"])
                print(valueset[:, 0].mean(), valueset[:, 0].std())
                print(valueset[:, 1].mean(), valueset[:, 1].std())
                print(valueset[:, 2].mean(), valueset[:, 2].std())

            rx = list(nsplit(range(len(smileset)), comm_size))[rank]
            info("subset range:", idataset, len(smileset), rx.start, rx.stop)
            ## local portion
            _smileset = smileset[rx.start : rx.stop]
            _valueset = valueset[rx.start : rx.stop]
            info("local smileset size:", len(_smileset))

            for smilestr, ytarget in iterate_tqdm(
                zip(_smileset, _valueset), verbosity, total=len(_smileset)
            ):
                data = generate_graphdata(smilestr, ytarget, var_config)
                dataset_lists[idataset].append(data)

        ## local data
        _trainset = dataset_lists[0]
        _valset = dataset_lists[1]
        _testset = dataset_lists[2]

        adwriter = AdiosOGB("examples/ogb/dataset/ogb_gap.bp", comm)
        adwriter.add("trainset", _trainset)
        adwriter.add("valset", _valset)
        adwriter.add("testset", _testset)
        adwriter.save()

        sys.exit(0)

    gp.initialize()
    timer = Timer("load_data")
    timer.start()
    opt = {"preload": True}
    if not args.noadios:
        if args.shmem:
            trainset = OGBDataset(
                "examples/ogb/dataset/ogb_gap.bp",
                "trainset",
                comm,
                preload=False,
                shmem=True,
            )
        else:
            trainset = OGBDataset(
                "examples/ogb/dataset/ogb_gap.bp", "trainset", comm, opt
            )
        valset = OGBDataset("examples/ogb/dataset/ogb_gap.bp", "valset", comm, opt)
        testset = OGBDataset("examples/ogb/dataset/ogb_gap.bp", "testset", comm, opt)
    else:
        fact = OGBRawDatasetFactory(
            "examples/ogb/dataset/pcqm4m_gap.csv",
            var_config=var_config,
            sampling=args.sampling,
        )
        trainset = OGBRawDataset(fact, "trainset")
        valset = OGBRawDataset(fact, "valset")
        testset = OGBRawDataset(fact, "testset")

    info("Adios load")
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    if rank == 0:
        print_model(model)
    dist.barrier()

    learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=True,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    gp.pr_file("./logs/%s/gp_timing.%d" % (log_name, rank))
    gp.pr_summary_file("./logs/%s/gp_timing.summary" % (log_name))
    gp.finalize()

    if args.mae:
        ##################################################################################################################
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        isub = -1
        for loader, setname in zip(
            [train_loader, val_loader, test_loader], ["train", "val", "test"]
        ):
            error, rmse_task, true_values, predicted_values = hydragnn.train.test(
                loader, model, verbosity
            )
            ihead = 0
            head_true = np.asarray(true_values[ihead].cpu()).squeeze()
            head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = graph_feature_names[ifeat]

            isub += 1
            ax = axs[isub]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            print(varname, ": ev, mae=", error_mae, ", rmse= ", error_rmse)
            print(rmse_task[ihead])
            print(head_pred.shape, head_true.shape)

            ax.scatter(
                head_true,
                head_pred,
                s=7,
                linewidth=0.5,
                edgecolor="b",
                facecolor="none",
            )
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname + " (eV)", fontsize=16)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2f}".format(error_mae),
            )
        if rank == 0:
            fig.savefig("./logs/" + log_name + "/" + varname + "_all.png")
        plt.close()

    if args.shmem:
        trainset.unlink()

    sys.exit(0)

    ##################################################################################################################
    for ifeat in range(len(var_config["output_index"])):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        plt.subplots_adjust(
            left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
        )
        ax = axs[0]
        ax.scatter(
            range(len(trainset)),
            [trainset[i].y[ifeat].item() for i in range(len(trainset))],
            edgecolor="b",
            facecolor="none",
        )
        ax.set_title("train, " + str(len(trainset)))
        ax = axs[1]
        ax.scatter(
            range(len(valset)),
            [valset[i].y[ifeat].item() for i in range(len(valset))],
            edgecolor="b",
            facecolor="none",
        )
        ax.set_title("validate, " + str(len(valset)))
        ax = axs[2]
        ax.scatter(
            range(len(testset)),
            [testset[i].y[ifeat].item() for i in range(len(testset))],
            edgecolor="b",
            facecolor="none",
        )
        ax.set_title("test, " + str(len(testset)))
        fig.savefig(
            "./logs/"
            + log_name
            + "/ogb_train_val_test_"
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
            [
                item
                for i in range(len(valset))
                for item in valset[i].x[:, ifeat].tolist()
            ],
            "bo",
        )
        ax.set_title("validate, " + str(len(valset)))
        ax = axs[2]
        ax.plot(
            [
                item
                for i in range(len(testset))
                for item in testset[i].x[:, ifeat].tolist()
            ],
            "bo",
        )
        ax.set_title("test, " + str(len(testset)))
        fig.savefig(
            "./logs/"
            + log_name
            + "/ogb_train_val_test_"
            + var_config["input_node_feature_names"][ifeat]
            + ".png"
        )
        plt.close()
