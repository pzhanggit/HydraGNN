import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.utils.atomicdescriptors import atomicdescriptors
from hydragnn.utils.print_utils import print_distributed
from hydragnn.preprocess.raw_dataset_loader import RawDataLoader
from hydragnn.utils.model import print_model, load_existing_model
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import mendeleev
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np
from scipy.interpolate import griddata
import joblib
from autoencoder import *

def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def update_graphdata_from_smilestr(smiles, data):

    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    
    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    mol = Chem.MolFromSmiles(smiles, ps)  
    mol = Chem.AddHs(mol)
    N = mol.GetNumAtoms()

    # create a 3D conformer:
    embedflag = rdDistGeom.EmbedMolecule(mol, randomSeed=20)
    assert embedflag==0, "Embedding fails for %d %s"%(data.sample_id, data.smiles)
    conf = mol.GetConformer()


    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type, bond_length = [], [], [],[]
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        bond_length += 2 * [rdMolTransforms.GetBondLength(conf, start, end)]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    #edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
    bond_length = torch.tensor(bond_length)
    edge_type = torch.tensor(edge_type)
    edge_attr = torch.stack((bond_length, edge_type), dim=1)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x = (
        torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float)
        .t()
        .contiguous()
    )
    data.x = x ##z.reshape((-1, 1)) #x
    data.edge_index = edge_index
    data.edge_attr = edge_attr

    return data 

def getcolordensity(xdata, ydata):
    ###############################
    print("###############################")
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=xdata, y=ydata, bins=[nbin, nbin]
    )
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)

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
    ) 
    return hist2d_norm
class myatomicdescriptors(atomicdescriptors):
    def __init__(
        self,
        embeddingfilename,
        overwritten=True,
        element_types=["C", "H", "O", "N","F"],
        one_hot=False,
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
            self.one_hot = one_hot
            type_id = self.get_type_ids()
            group_id = self.get_group_ids()
            period = self.get_period()
            covalent_radius = self.get_covalent_radius()
            electron_affinity = self.get_electron_affinity()
            block = self.get_block()
            atomic_volume = self.get_atomic_volume()
            atomic_number = self.get_atomic_number()
            atomic_weight = self.get_atomic_weight()
            electronegativity = self.get_electronegativity()
            valenceelectrons = self.get_valence_electrons()
            ionenergies = self.get_ionenergies()
            if self.one_hot:
                # properties with integer values
                group_id = self.convert_integerproperty_onehot(group_id, num_classes=-1)
                period = self.convert_integerproperty_onehot(period, num_classes=-1)
                atomic_number = self.convert_integerproperty_onehot(
                    atomic_number, num_classes=-1
                )
                valenceelectrons = self.convert_integerproperty_onehot(
                    valenceelectrons, num_classes=-1
                )
                # properties with real values
                covalent_radius = self.convert_realproperty_onehot(
                    covalent_radius, num_classes=10
                )
                electron_affinity = self.convert_realproperty_onehot(
                    electron_affinity, num_classes=10
                )
                atomic_volume = self.convert_realproperty_onehot(
                    atomic_volume, num_classes=10
                )
                atomic_weight = self.convert_realproperty_onehot(
                    atomic_weight, num_classes=10
                )
                electronegativity = self.convert_realproperty_onehot(
                    electronegativity, num_classes=10
                )
                ionenergies = self.convert_realproperty_onehot(
                    ionenergies, num_classes=10
                )

            for iele, ele in enumerate(self.element_types):
                nfeatures = 0
                self.atom_embeddings[str(mendeleev.element(ele).atomic_number)] = []
                for var in [
                    type_id,
                    covalent_radius,
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
        "--normalizey",
        action="store_true",
        help="normalize spectrum for each sample to be [0, 1]",
    )
    parser.add_argument(
        "--pca",
        type=int,
        default=0,
        help="number of output PCs",
    )
    parser.add_argument(
        "--startline",
        type=int,
        default=500,
        help="start line position in spectrum component file",
    )
    parser.add_argument(
        "--endline",
        type=int,
        default=1001,
        help="end line position in spectrum component file",
    )
    parser.add_argument(
        "--loadsplit",
        action="store_true",
        help="load the same splits",
    )
    parser.add_argument(
        "--inputfile",
        help="input file",
        type=str,
        default="vibrational_spectroscopy.json",
    )
    parser.add_argument(
        "--logname",
        help="output directory",
        type=str,
        default="qm8",
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
    
    log_name = args.logname #get_log_name_config(config)
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
        #############test outputsize###############################
        config["Dataset"]["textposition"]=[args.startline, args.endline]
        ## each process saves its own data file
        loader = RawDataLoader(config["Dataset"], dist=True)
        loader.load_raw_data()
        
        if args.loadsplit:
            pathcase=config["Dataset"]["path"]["total"]
            config["Dataset"]["splitfile"]=os.path.join(dirpwd, "./dataset/%s_splitlist.pk"%pathcase[pathcase.rfind("/")+1:])
    
        ## Read total pkl and split (no graph object conversion)
        hydragnn.preprocess.total_to_train_val_test_pkls(config, isdist=True)

        ## Read each pkl and graph object conversion with max-edge normalization
        (
            trainset,
            valset,
            testset,
        ) = hydragnn.preprocess.load_data.load_train_val_test_sets(config, isdist=True)


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
    if "edge_connectivity" in config["Dataset"] and config["Dataset"]["edge_connectivity"]=="SMILES":
        for dataset in [trainset, valset, testset]:
            print("==============================================================")
            for graphdata in dataset:
                #print(graphdata.sample_id, graphdata.smiles)
                graphdata = update_graphdata_from_smilestr(graphdata.smiles, graphdata)
                #print(graphdata.x)
                print(graphdata.x.size(), graphdata.edge_index.size(), graphdata.edge_attr.size(), graphdata.y.size())
        config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]=[*range(graphdata.x.size()[1])]
        config["NeuralNetwork"]["Architecture"]["edge_features"] = ["length","bond_type"]
    if args.mendeleev:
        if "data_source" in config["Dataset"] and config["Dataset"]["data_source"]=="mixed":
            atomicdescriptor = myatomicdescriptors("./embedding_mendeleev_test_mixed.json", overwritten=True, element_types=["C", "H", "O", "N", "F","P", "S", "Cl", "Si", "Br", "I"])
        else:
            atomicdescriptor = myatomicdescriptors("./embedding_mendeleev_test.json", overwritten=True, element_types=["C", "H", "O", "N", "F"])
        #unknown=[]
        for dataset in [trainset, valset, testset]:
            for graphdata in dataset:
                xfeature = []
                #print(graphdata.x)
                #print(graphdata.num_of_protons)
                for inode in range(graphdata.num_nodes):
         #           try:
                    xfeature.append(atomicdescriptor.get_atom_features(graphdata.num_of_protons[inode].item()))
                    #except:
                    #    if graphdata.num_of_protons[inode].item() not in unknown:
                    #        unknown.append(graphdata.num_of_protons[inode].item())
                graphdata.x = torch.stack(xfeature, dim=1).transpose(0,1)
                print(graphdata.x.size())
        #print("unknown elments, ", unknown)
        config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]=[*range(graphdata.x.size()[1])]
    if args.normalizey:
       mol_nan = []
       for dataset in [trainset, valset, testset]:
           for graphdata in dataset:
               #print(graphdata.y.min(), graphdata.y.max())
               #if graphdata.y.max() > graphdata.y.min():
                   #graphdata.y = (graphdata.y - graphdata.y.min())/(graphdata.y.max()-graphdata.y.min())
               ymin = graphdata.y.view(-1, config["Dataset"]["graph_features"]["dim"][0]).min(1)[0]
               ymax = graphdata.y.view(-1, config["Dataset"]["graph_features"]["dim"][0]).max(1)[0]
               #print(graphdata.y.size())
               #print(ymin, ymax)
               #print(ymin.size(), ymax.size())
               #print(ymin.reshape((-1,1)).size(), ymax.reshape((-1,1)).size())
               if torch.all(ymax > ymin):
                   ymax_all = ymax.repeat_interleave(config["Dataset"]["graph_features"]["dim"][0]).reshape((-1,1))
                   ymin_all = ymin.repeat_interleave(config["Dataset"]["graph_features"]["dim"][0]).reshape((-1,1))
                   graphdata.y = (graphdata.y - ymin_all)/(ymax_all-ymin_all)
               #print("==============================================================")
               if torch.isnan(graphdata.y).any():
                 #  print(graphdata.sample_id, graphdata.y)
                   mol_nan.append(str(graphdata.sample_id))
               #print(graphdata.y.min(), graphdata.y.max())
               #print("==============================================================")
       assert len(mol_nan)==0, "nan after normalization for samples: "+', '.join(mol_nan)
    if args.pca>0:
       npca=args.pca
       if datasetname == "qm8":
           pcafile=dirpwd+"/../../../test/SiC_Chemistry_Reduction/logs/nz-%d-PCA-True/PCA.joblib"%npca
       else:    
           pcafile=dirpwd+"/../../../test/SiC_Chemistry_Reduction/logs_QM9-DFT-HQ/nz-%d-PCA-True/PCA.joblib"%npca
       pca=joblib.load(pcafile)
       for dataset in [trainset, valset, testset]:
          for graphdata in dataset:
              graphdata.y0=graphdata.y
              zpca=pca.transform(graphdata.y.transpose(0, 1))
              graphdata.y=torch.from_numpy(zpca).type(graphdata.y.dtype).transpose(0, 1)
              assert graphdata.y.shape[0]==npca, graphdata.y
    elif args.pca<0:
       npca=-args.pca
       ae_model = autoencoder(501, reduced_dim=npca, hidden_dim_ae=[250], PCA=False)
       #modelfile=dirpwd+"/../../../test/SiC_Chemistry_Reduction/logs/nz-%d-PCA-False/model.pk"%npca
       if datasetname == "qm8":
           modelfile=dirpwd+"/../../../test/SiC_Chemistry_Reduction/logs/nz-%d-PCA-False/model.pk"%npca
       else:    
           modelfile=dirpwd+"/../../../test/SiC_Chemistry_Reduction/logs_QM9-DFT-HQ/nz-%d-PCA-False/model.pk"%npca
       print("Load existing model:", modelfile)
       ae_model.load_state_dict(torch.load(modelfile))
       ae_model.eval()
       with torch.no_grad():
          for dataset in [trainset, valset, testset]:
             for graphdata in dataset:
                 graphdata.y0=graphdata.y
                 zpca=ae_model.encoder(graphdata.y.transpose(0, 1))
                 graphdata.y=zpca.transpose(0, 1)
                 assert graphdata.y.shape[0]==npca, graphdata.y

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    if args.pca != 0:
       config["NeuralNetwork"]["Architecture"]["output_dim"] = [npca]

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
    weight_decay = 0.01 #the default value is 0.01 in pytorch for AdamW
    if "weight_decay" in config["NeuralNetwork"]["Training"]["Optimizer"]:
        weight_decay = config["NeuralNetwork"]["Training"]["Optimizer"]["weight_decay"] 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    #assert log_name == get_log_name_config(config), f"Expect {log_name}, but get {get_log_name_config(config)}"
    print(f"Expect {log_name}, but get {get_log_name_config(config)}")
    writer = hydragnn.utils.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()
    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)

    plot_init_solution = config["Visualization"]["plot_init_solution"]
    plot_hist_solution = config["Visualization"]["plot_hist_solution"]
    create_plots = config["Visualization"]["create_plots"]
    for param_group in optimizer.param_groups:
        print(param_group)
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
        plot_init_solution=True,#plot_init_solution,
        plot_hist_solution=plot_hist_solution,
        create_plots=True,#create_plots,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    for isub, (loader, setname) in enumerate(
       zip([train_loader, val_loader, test_loader], ["train", "val", "test"])
    ):
        (
            test_rmse,
            test_taskserr,
            true_values,
            predicted_values,
            sample_ids,
        ) = hydragnn.train.test(loader, model, verbosity, return_sampleid=True)
        num_samples = int(len(true_values[0]) / model.module.head_dims[0])
        print(f"num_samples in test_set={num_samples}")
        if rank == 0:
            output_dir = f"./logs/{log_name}/spectrum/{setname}"
            os.makedirs(output_dir, exist_ok=True)
            ###########################################################  
            if args.pca != 0 and setname=="test":
                varname = config["NeuralNetwork"]["Variables_of_interest"]["output_names"][0]
                output_dir_all = f"./logs/{log_name}/spectrum/{setname}/true_all"
                os.makedirs(output_dir_all, exist_ok=True)
                for graphdata in loader.dataset:
                    textfile = open(os.path.join(output_dir_all,varname
                        + "_true_value_" + str(graphdata.sample_id)+ ".txt"),"w+")
                    for element in graphdata.y0:
                        textfile.write(str(element.item()) + "\n")
                    textfile.close()
            ###########################################################        
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
                            varname + "_" + str(sample_ids[isample].item()) + "_Zs.png",
                        )
                    )
                    plt.close()
            ###########################################################        
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
                if args.pca>0:
                    head_true = torch.from_numpy(pca.inverse_transform(head_true.to("cpu")))
                    head_pred = torch.from_numpy(pca.inverse_transform(head_pred.to("cpu")))
                elif args.pca<0:
                    head_true = ae_model.decoder(head_true.to("cpu")).detach()
                    head_pred = ae_model.decoder(head_pred.to("cpu")).detach()
                head_true_np = np.asarray(head_true.cpu()).squeeze()
                head_pred_np = np.asarray(head_pred.cpu()).squeeze()
                error_mae = np.mean(np.abs(head_pred_np - head_true_np))
                error_rmse = np.sqrt(np.mean(np.abs(head_pred_np - head_true_np) ** 2))
                print(varname,", ", setname, ": , mae=", error_mae, ", rmse= ", error_rmse)
                print_distributed(verbosity, varname,", ", setname, ": , mae=", error_mae, ", rmse= ", error_rmse)
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
    ##################################################################################################################
    print("Before final parity plot")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for isub, (loader, setname) in enumerate(
       zip([train_loader, val_loader, test_loader], ["train", "val", "test"])
    ):
        (
          test_rmse,
          test_taskserr,
          true_values,
          predicted_values,
          sample_ids,
        ) = hydragnn.train.test(loader, model, verbosity, return_sampleid=True)
        num_samples = int(len(true_values[0]) / model.module.head_dims[0])
        print(f"num_samples in {setname} = {num_samples}")
        ihead = 0
        head_true = np.asarray(true_values[ihead].cpu()).squeeze()
        head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
        varname = config["NeuralNetwork"]["Variables_of_interest"]["output_names"][
            ihead
        ]

        ax = axs[isub]
        error_mae = np.mean(np.abs(head_pred - head_true))
        error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
        print(varname, ", Zs, ", setname,": , mae=", error_mae, ", rmse= ", error_rmse)
        print_distributed(verbosity, varname,", Zs, ", setname, ": , mae=", error_mae, ", rmse= ", error_rmse)
        hist2d_norm = getcolordensity(head_true, head_pred)
        ax.scatter(head_true, head_pred, s=7, c=hist2d_norm, vmin=0, vmax=1)
        #ax.scatter(
        #   head_true,
        #   head_pred,
        #   s=7,
        #   linewidth=0.5,
        #   edgecolor="b",
        #   facecolor="none",
        #)
        minv = np.minimum(np.amin(head_pred), np.amin(head_true))
        maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
        ax.plot([minv, maxv], [minv, maxv], "r--")
        ax.set_title(setname + "; " + varname, fontsize=16)
        ax.text(
           minv + 0.1 * (maxv - minv),
           maxv - 0.1 * (maxv - minv),
           "MAE: {:.2f}".format(error_mae),
        )
    if rank == 0:
        fig.savefig("./logs/" + log_name + "/" + varname + "_all.png")
    plt.close()

    sys.exit(0)
