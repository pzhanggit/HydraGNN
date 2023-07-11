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

from torch_geometric.data import Data
from torch import tensor

from hydragnn.preprocess.raw_dataset_loader import AbstractRawDataLoader

from rdkit import Chem
from rdkit.Chem import rdDistGeom
import math
import os
# WARNING: DO NOT use collective communication calls here because only rank 0 uses this routines


class YQ_RawDataLoader(AbstractRawDataLoader):
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data()
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    def __init__(self, config, dist=False):
        super(YQ_RawDataLoader, self).__init__(config, dist)

    def transform_input_to_data_object_base(self, filepath):
        data_object = self.__transform_spectroscopy_input_to_data_object_base(
                filepath=filepath
            )

        return data_object

    def __transform_spectroscopy_input_to_data_object_base(self, filepath):
        """Transforms lines of strings read from the raw data CFG file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        if filepath.endswith(".xyz"):

            data_object = self.__transform_YQ_object_to_data_object(filepath)

            return data_object

        else:
            return None

    def __transform_YQ_object_to_data_object(self, filepath):
        """Transforms lines of strings read from the raw data file to Data object and returns it.
        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """
        #check if files exist
        if not filepath.endswith(".xyz"):
            return None
        filename_without_extension = os.path.splitext(filepath)[0]
        filedir = filename_without_extension.rsplit("/", 1)[0]
        index = filename_without_extension.rsplit("/", 1)[1]
        if not os.path.exists(filename_without_extension + "_vis_inc_0K.csv"):
            return None
        if self.edge_connectivity is not None:
            if not os.path.exists(filedir+"/INFO-"+index+".dat"):
                return None
        
        data_object = Data()

        # input files
        if filepath.find("xyz") != -1:
            f = open(filepath, "r", encoding="utf-8")

            node_feature_matrix = []
            node_position_matrix = []
            num_of_protons = []
            all_lines = f.readlines()

            for line in all_lines:
                node_feat = line.split(None, 11)

                x_pos = float(node_feat[1].strip())
                y_pos = float(node_feat[2].strip())
                z_pos = float(node_feat[3].strip())
                node_position_matrix.append([x_pos, y_pos, z_pos])
                num_of_protons.append(int(node_feat[0].strip()))

                node_feature = []
                node_feature_dim = [1]
                node_feature_col = [0]
                for item in range(len(node_feature_dim)):
                    for icomp in range(node_feature_dim[item]):
                        it_comp = node_feature_col[item] + icomp
                        node_feature.append(float(node_feat[it_comp].strip()))
                node_feature_matrix.append(node_feature)

            data_object.pos = tensor(node_position_matrix)
            data_object.x = tensor(node_feature_matrix)
            #data_object.x = torch.nn.functional.one_hot(
            #    data_object.x.view(-1).to(torch.int64), num_classes=118
            #)
            data_object.num_of_protons = tensor(num_of_protons)
        #filename_without_extension = os.path.splitext(filepath)[0]
        #index = filename_without_extension.rsplit("/", 1)[1]
        data_object.sample_id = int(index)

        # output files
        if os.path.exists(filename_without_extension + "_vis_inc_0K" + ".csv") != -1:

            f = open(
                filename_without_extension + "_vis_inc_0K" + ".csv",
                "r",
                encoding="utf-8",
            )
            all_lines = f.readlines()

            g_feature = []

            start_line = self.startline #500
            n_features = 11
            n_partial= math.ceil((self.endline-self.startline)/self.graph_feature_dim[0])
            #end_line = start_line + self.graph_feature_dim[0]
            #end_line = start_line + 2*self.graph_feature_dim[0]
            end_line = start_line + n_partial*self.graph_feature_dim[0]

            #for line in all_lines[start_line:end_line]:
            #for line in all_lines[start_line:end_line:2]:
            for line in all_lines[start_line:end_line:n_partial]:
                list_feat = line.split(",", n_features)
                for icol in self.graph_feature_col:
                    if isinstance(icol, int):
                        g_feature.append(float(list_feat[icol]))
                    elif isinstance(icol, list):
                        g_feature.append(sum(float(list_feat[item]) for item in icol))

            data_object.y = (
                tensor(g_feature)
                .reshape([-1, len(self.graph_feature_dim)])
                .transpose(0, 1)
                .flatten()
            )
            print(data_object.y.size())
        if self.edge_connectivity=="SMILES":
            with open(filedir+"/INFO-"+index+".dat") as f:
                data_object.smiles = f.readline().strip('\n')
            #prechecking 
            ps = Chem.SmilesParserParams()
            ps.removeHs = False
            mol = Chem.MolFromSmiles(data_object.smiles, ps)  
            mol = Chem.AddHs(mol)
            embedflag = rdDistGeom.EmbedMolecule(mol, randomSeed=20)
            if embedflag ==-1:
                print("Embedding failed for: %d %s"%(data_object.sample_id,  data_object.smiles))
                return None

        return data_object
