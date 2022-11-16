import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.utils.atomicdescriptors import atomicdescriptors
from hydragnn.preprocess.raw_dataset_loader import RawDataLoader
from hydragnn.utils.model import print_model
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import mendeleev

def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

class myatomicdescriptors(atomicdescriptors):
    def __init__(
        self,
        embeddingfilename,
        overwritten=True,
        element_types=["C", "H", "O", "N","F"],
    ):
        if os.path.exists(embeddingfilename) and not overwritten:
            print("loading from existing file: ", embeddingfilename)
            with open(embeddingfilename, "r") as f:
                self.atom_embeddings = json.load(f)
        else:
            self.atom_embeddings = {}
            if element_types is None:
                self.element_types = []
                for ele in mendeleev.get_all_elements():
                    self.element_types.append(ele.symbol)
            else:
                self.element_types = []
                for ele in mendeleev.get_all_elements():
                    if ele.symbol in element_types:
                        self.element_types.append(ele.symbol)
                self.element_types = element_types
            electron_affinity = self.get_electron_affinity()
            atomic_volume = self.get_atomic_volume()
            atomic_number = self.get_atomic_number()
            atomic_weight = self.get_atomic_weight()
            for iele, ele in enumerate(self.element_types):
                nfeatures = 0
                self.atom_embeddings[str(mendeleev.element(ele).atomic_number)] = []
                for var in [
                    electron_affinity,
                    atomic_volume,
                    atomic_number,
                    atomic_weight,
                ]:
                    nfeatures += var.size()[1]
                    self.atom_embeddings[
                        str(mendeleev.element(ele).atomic_number)
                    ].extend(var[iele, :].tolist())
            with open(embeddingfilename, "w") as f:
                json.dump(self.atom_embeddings, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loadexistingsplit",
        action="store_true",
        help="loading from existing pickle/adios files with train/test/validate splits",
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios or pickle saving and no train",
    )
    parser.add_argument(
        "--mendeleev",
        action="store_true",
        help="use atomic descriptors from mendeleev",
    )
    parser.add_argument(
        "--inputfile",
        help="input file",
        type=str,
        default="vibrational_spectroscopy.json",
    )
    parser.add_argument(
        "--testplotskip",
        help="skip samples in spectrum plot if >1",
        type=int,
        default=1,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(__file__)
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    #hydragnn.utils.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    os.environ["SERIALIZED_DATA_PATH"] = dirpwd + "/dataset"
    datasetname = config["Dataset"]["name"]
    fname_adios = dirpwd + "/dataset/%s.bp" % (datasetname)
    config["Dataset"]["name"] = "%s_%d" % (datasetname, rank)
    
    log_name = get_log_name_config(config)
    hydragnn.utils.setup_log(log_name)
    comm = MPI.COMM_WORLD
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )
    
    if not args.loadexistingsplit:
        for dataset_type, raw_data_path in config["Dataset"]["path"].items():
            if not os.path.isabs(raw_data_path):
                raw_data_path = os.path.join(dirpwd, raw_data_path)
            if not os.path.exists(raw_data_path):
                raise ValueError("Folder not found: ", raw_data_path)
            config["Dataset"]["path"][dataset_type] = raw_data_path

        ## each process saves its own data file
        loader = RawDataLoader(config["Dataset"], dist=True)
        loader.load_raw_data()

        ## Read total pkl and split (no graph object conversion)
        hydragnn.preprocess.total_to_train_val_test_pkls(config, isdist=True)

        ## Read each pkl and graph object conversion with max-edge normalization
        (
            trainset,
            valset,
            testset,
        ) = hydragnn.preprocess.load_data.load_train_val_test_sets(config, isdist=True)

        if args.mendeleev:
            atomicdescriptor = myatomicdescriptors("./embedding_mendeleev.json", overwritten=False, element_types=["C", "H", "O", "N", "F"])
            for dataset in [trainset, valset, testset]:
                for graphdata in dataset:
                    xfeature = []
                    #print(graphdata.x)
                    #print(graphdata.num_of_protons)
                    for inode in range(graphdata.num_nodes):
                        xfeature.append(atomicdescriptor.get_atom_features(graphdata.num_of_protons[inode].item()))
                    graphdata.x = torch.stack(xfeature, dim=1).transpose(0,1)
                    print(graphdata.x.size())

        if args.format == "adios":
            from hydragnn.utils.adiosdataset import AdiosWriter

            adwriter = AdiosWriter(fname_adios, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.add_global("minmax_node_feature", loader.minmax_node_feature)
            adwriter.add_global("minmax_graph_feature", loader.minmax_graph_feature)
            adwriter.save()
    if args.preonly:
        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        from hydragnn.utils.adiosdataset import AdiosDataset

        info("Adios load")
        trainset = AdiosDataset(fname_adios, "trainset", comm)
        valset = AdiosDataset(fname_adios, "valset", comm)
        testset = AdiosDataset(fname_adios, "testset", comm)
        ## Set minmax read from bp file
        config["NeuralNetwork"]["Variables_of_interest"][
            "minmax_node_feature"
        ] = trainset.minmax_node_feature
        config["NeuralNetwork"]["Variables_of_interest"][
            "minmax_graph_feature"
        ] = trainset.minmax_graph_feature
    elif args.format == "pickle":
        config["Dataset"]["path"] = {}
        ##set directory to load processed pickle files, train/validate/test
        for dataset_type in ["train", "validate", "test"]:
            raw_data_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}_{dataset_type}.pkl"
            config["Dataset"]["path"][dataset_type] = raw_data_path
        info("Pickle load")
        (
            trainset,
            valset,
            testset,
        ) = hydragnn.preprocess.load_data.load_train_val_test_sets(config, isdist=True)
        # FIXME: here is a navie implementation with allgather. Need to have better/faster implementation
        trainlist = [None for _ in range(dist.get_world_size())]
        vallist = [None for _ in range(dist.get_world_size())]
        testlist = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(trainlist, trainset)
        dist.all_gather_object(vallist, valset)
        dist.all_gather_object(testlist, testset)
        trainset = [item for sublist in trainlist for item in sublist]
        valset = [item for sublist in vallist for item in sublist]
        testset = [item for sublist in testlist for item in sublist]
    else:
        raise ValueError("Unknown data format: %d" % args.format)

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    assert log_name == get_log_name_config(config), f"Expect {log_name}, but get {get_log_name_config(config)}"
    writer = hydragnn.utils.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()
    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)

    plot_init_solution = config["Visualization"]["plot_init_solution"]
    plot_hist_solution = config["Visualization"]["plot_hist_solution"]
    create_plots = config["Visualization"]["create_plots"]
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
        plot_init_solution=plot_init_solution,
        plot_hist_solution=plot_hist_solution,
        create_plots=create_plots,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    (
        test_rmse,
        test_taskserr,
        true_values,
        predicted_values,
        sample_ids,
    ) = hydragnn.train.test(test_loader, model, verbosity, return_sampleid=True)
    num_samples = int(len(true_values[0]) / model.module.head_dims[0])
    print(f"num_samples in test_set={num_samples}")
    if rank == 0:
        output_dir = f"./logs/{log_name}/spectrum"
        os.makedirs(output_dir, exist_ok=True)
        for ihead in range(model.module.num_heads):
            varname = config["NeuralNetwork"]["Variables_of_interest"]["output_names"][
                ihead
            ]
            head_true = torch.reshape(
                true_values[ihead], (-1, model.module.head_dims[ihead])
            )
            head_pred = torch.reshape(
                predicted_values[ihead], (-1, model.module.head_dims[ihead])
            )
            for isample in range(0, num_samples, args.testplotskip):
                print(isample)
                plt.figure()
                plt.plot(head_true[isample, :].to("cpu"))
                plt.plot(head_pred[isample, :].to("cpu"))
                plt.title(sample_ids[isample].item())
                plt.draw()
                plt.savefig(
                    os.path.join(
                        output_dir,
                        varname + "_" + str(sample_ids[isample].item()) + ".png",
                    )
                )
                plt.close()

                textfile = open(
                    os.path.join(
                        output_dir,
                        varname
                        + "_true_value_"
                        + str(sample_ids[isample].item())
                        + ".txt",
                    ),
                    "w+",
                )
                for element in head_true[isample, :]:
                    textfile.write(str(element.item()) + "\n")
                textfile.close()

                textfile = open(
                    os.path.join(
                        output_dir,
                        varname
                        + "_predicted_value_"
                        + str(sample_ids[isample].item())
                        + ".txt",
                    ),
                    "w+",
                )
                for element in head_pred[isample, :]:
                    textfile.write(str(element.item()) + "\n")
                textfile.close()
    sys.exit(0)
