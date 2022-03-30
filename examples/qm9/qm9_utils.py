import sys, os
import torch
import pickle, csv

#########################################################
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
import hydragnn
from hydragnn.utils.distributed import get_comm_size_and_rank
##################################################################################################################
##################################################################################################################
HAR2EV = 27.211386246


def get_splitlists(filedir):
    with open(filedir, "rb") as f:
        train_filelist = pickle.load(f)
        val_filelist = pickle.load(f)
        test_filelist = pickle.load(f)
    return train_filelist, val_filelist, test_filelist


def get_trainset_stat(filedir):
    with open(filedir, "rb") as f:
        max_feature = pickle.load(f)
        min_feature = pickle.load(f)
        mean_feature = pickle.load(f)
        std_feature = pickle.load(f)
    return max_feature, min_feature, mean_feature, std_feature


def datasets_load_gap(datafile):
    smiles = []
    yvals = []
    with open(datafile, "r") as file:
        csvreader = csv.reader(file)
        print(next(csvreader))
        for row in csvreader:
            smiles.append(row[1])
            yvals.append(float(row[-1]) * HAR2EV)
    return smiles, yvals


def splits_save(datafile, splitlistfile):
    fileid_list = []
    values_list = []
    with open(datafile, "r") as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            fileid_list.append(row[0])
    perc_train = 0.9
    perc_val = (1 - perc_train) / 2
    ntotal = len(fileid_list)
    ntrain = int(ntotal * perc_train)
    nval = int(ntotal * perc_val)
    ntest = ntotal - ntrain - nval
    print(ntotal, ntrain, nval, ntest)
    randomlist = torch.randperm(ntotal)

    idx_train_list = [fileid_list[ifile] for ifile in randomlist[:ntrain]]
    idx_val_list = [fileid_list[ifile] for ifile in randomlist[ntrain : ntrain + nval]]
    idx_test_list = [fileid_list[ifile] for ifile in randomlist[ntrain + nval :]]

    _, rank = get_comm_size_and_rank()
    if rank == 0:
        with open(splitlistfile, "wb") as f:
            pickle.dump(idx_train_list, f)
            pickle.dump(idx_val_list, f)
            pickle.dump(idx_test_list, f)

    return idx_train_list, idx_val_list, idx_test_list


def datasets_load(datafile, splitlistfile):
    if os.path.isfile(splitlistfile):
        train_filelist, val_filelist, test_filelist = get_splitlists(splitlistfile)
    else:
        print("Generate new split since file not found: ", splitlistfile)
        train_filelist, val_filelist, test_filelist = splits_save(
            datafile, splitlistfile
        )

    trainset = []
    valset = []
    testset = []
    trainsmiles = []
    valsmiles = []
    testsmiles = []
    trainidxs = []
    validxs = []
    testidxs = []
    with open(datafile, "r") as file:
        csvreader = csv.reader(file)
        print(next(csvreader))
        for row in csvreader:
            if row[0] in train_filelist:
                trainsmiles.append(row[1])
                trainset.append([float(x) * HAR2EV for x in row[2:]])
                trainidxs.append(int(row[0][4:]))
            elif row[0] in val_filelist:
                valsmiles.append(row[1])
                valset.append([float(x) * HAR2EV for x in row[2:]])
                validxs.append(int(row[0][4:]))
            elif row[0] in test_filelist:
                testsmiles.append(row[1])
                testset.append([float(x) * HAR2EV for x in row[2:]])
                testidxs.append(int(row[0][4:]))
            else:
                print("unknown file name: ", row[0])
                sys.exit(0)
    return (
        [trainsmiles, valsmiles, testsmiles],
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)],
        [trainidxs, validxs, testidxs],
    )


##################################################################################################################
node_attribute_names = [
    "atomH",
    "atomC",
    "atomN",
    "atomO",
    "atomF",
    "atomicnumber",
    "IsAromatic",
    "HSP",
    "HSP2",
    "HSP3",
    "Hprop",
]


def gapfromsmiles(smilestr, model):
    ##idx and gap_true can be replaced by random numbers when use
    idx = 5700
    gap_rand = 0.0
    data_graph = generate_graphdata(idx, smilestr, gap_rand)
    pred = model(data_graph)
    return pred[0][0].item()


def generate_graphdata(idx, simlestr, ytarget, var_config=None):
    types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    mol = Chem.MolFromSmiles(simlestr, ps)  # , sanitize=False , removeHs=False)
    mol = Chem.AddHs(mol)
    N = mol.GetNumAtoms()

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = (
        torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float)
        .t()
        .contiguous()
    )

    x = torch.cat([x1.to(torch.float), x2], dim=-1)
    # x = torch.tensor([atomic_number], dtype=torch.float).view(-1,1)
    # print(x)

    y = ytarget  # .squeeze()

    data = Data(x=x, z=z, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=idx)
    if var_config is not None:
        hydragnn.preprocess.update_predicted_values(
            var_config["type"],
            var_config["output_index"],
            var_config["graph_features_dim"],
            var_config["node_feature_dim"],
            data,
        )
    device = hydragnn.utils.get_device()
    return data.to(device)
