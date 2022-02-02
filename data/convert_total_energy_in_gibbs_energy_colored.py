import scipy.special
import math
import os
import shutil
import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle

plt.rcParams.update({"font.size": 18})


# function to return key for any value
def get_key(dictionary, val):

    for key, value in dictionary.items():
        if val == value:
            return key

    return None


def from_energy_to_formation_gibbs(path_to_dir, elements_list, temperature_kelvin=0):

    new_dataset_path = path_to_dir[:-1] + "_gibbs_energy/"

    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path)
    os.makedirs(new_dataset_path)

    Kb_joule_per_kelvin = 1.380649 * 1e-23
    conversion_joule_rydberg = 4.5874208973812 * 1e17
    Kb_rydberg_per_kelvin = Kb_joule_per_kelvin * conversion_joule_rydberg

    pure_elements_energy = dict()
    element_counter = dict()

    min_formation_enthalpy = float("Inf")
    max_formation_enthalpy = -float("Inf")

    total_energy_list = []
    linear_mixing_energy_list = []
    composition_list = []
    formation_enthalpy_list = []
    formation_gibbs_energy_list = []

    # Search for the configurations with pure elements and store their total energy
    for filename in os.listdir(path_to_dir):

        for atom_type in elements_list:
            element_counter[atom_type] = 0

        df = pandas.read_csv(path_to_dir + filename, header=None, nrows=1)
        energies = np.asarray([float(s) for s in df[0][0].split()])
        total_energy = energies[0]

        df = pandas.read_csv(path_to_dir + filename, header=None, skiprows=1)
        num_atoms = df.shape[0]
        for atom_index in range(0, num_atoms):
            row = df[0][atom_index].split()
            atom_type_str = row[0]
            element_counter[atom_type_str] += 1

        pure_element = get_key(element_counter, num_atoms)
        if pure_element is not None:
            pure_elements_energy[pure_element] = float(total_energy) / num_atoms

    # extract formation enthalpy from total energy
    # compute thermodynamic entropy
    # compute formation gibbs energy using formation enthalpy and thermodynamic entropy
    for filename in os.listdir(path_to_dir):

        (
            composition_element1,
            total_energy,
            linear_mixing_energy,
            formation_enthalpy,
        ) = compute_formation_enthalpy(
            path_to_dir + filename, elements_list, pure_elements_energy
        )

        # This is thermodynamic entropy, not statistical entropy
        # because we do not multiply the binomial coefficient by the probabilities
        entropy = Kb_rydberg_per_kelvin * math.log(
            scipy.special.comb(num_atoms, float(element_counter[elements_list[0]]))
        )

        formation_gibbs_energy = formation_enthalpy - temperature_kelvin * entropy

        min_formation_enthalpy = min(min_formation_enthalpy, formation_enthalpy)
        max_formation_enthalpy = max(max_formation_enthalpy, formation_enthalpy)

        total_energy_list.append(total_energy)
        linear_mixing_energy_list.append(linear_mixing_energy)
        composition_list.append(composition_element1)
        formation_enthalpy_list.append(formation_enthalpy)
        formation_gibbs_energy_list.append(formation_gibbs_energy)

        df = pandas.read_csv(path_to_dir + filename, header=None)
        df[0][0] = str(formation_gibbs_energy)
        df.to_csv(new_dataset_path + filename, header=None, index=None)

    print("Min formation enthalpy: ", min_formation_enthalpy)
    print("Max formation enthalpy: ", max_formation_enthalpy)
    #################################################
    with open("composition_mixingenthalpy.pkl", "wb") as f:
        pickle.dump(composition_list, f)
        pickle.dump(formation_enthalpy_list, f)
    #################################################
    comp_unq = list(set(composition_list))
    com_freq = [composition_list.count(comp_) for comp_ in comp_unq]
    # hist1d, xbins_edge = np.histogram(composition_list, bins=nbin)
    # xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])

    fig = plt.figure(figsize=(7, 6))
    # plt.plot(xbin_cen, hist1d)
    plt.plot(comp_unq, com_freq, "ro-")
    plt.xlabel("Fe concentration")
    plt.ylabel("Hist1d")
    plt.title("FePt: " + str(len(composition_list)))
    fig.subplots_adjust(
        left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.1, hspace=0.06
    )
    plt.savefig("histogram_Fe_concentration.png", dpi=400)
    ###############################
    nbin = 20
    xbins_edge = [comp_unq_ - 1.0 / 32.0 for comp_unq_ in comp_unq]
    xbins_edge.append(1 + 1.0 / 32.0)
    print(xbins_edge)
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=composition_list, y=formation_enthalpy_list, bins=[xbins_edge, nbin]
    )
    print(xbins_edge)

    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)

    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (composition_list, formation_enthalpy_list),
        method="linear",
        fill_value=0,
    )  # np.nan)
    #################################################
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(
        composition_list, formation_enthalpy_list, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("Fe concentration")
    plt.ylabel("Mixing enthalpy (Rydberg)")
    plt.title("FePt")
    fig.subplots_adjust(
        left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.1, hspace=0.06
    )
    plt.savefig("formation_enthalpy_colored.png", dpi=400)
    plt.savefig("formation_enthalpy_colored.pdf")


def plot_only():
    #################################################
    with open("composition_mixingenthalpy.pkl", "rb") as f:
        composition_list = pickle.load(f)
        formation_enthalpy_list = pickle.load(f)
    #################################################
    comp_unq = sorted(list(set(composition_list)))
    com_freq = [composition_list.count(comp_) for comp_ in comp_unq]
    # hist1d, xbins_edge = np.histogram(composition_list, bins=nbin)
    # xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])

    fig = plt.figure(figsize=(7, 6))
    # plt.plot(xbin_cen, hist1d)
    plt.plot(comp_unq, com_freq, "ro-")
    plt.xlabel("Fe concentration")
    plt.ylabel("Hist1d")
    plt.title("FePt: " + str(len(composition_list)))
    fig.subplots_adjust(
        left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.1, hspace=0.06
    )
    plt.savefig("histogram_Fe_concentration.png", dpi=400)
    ###############################
    nbin = 20
    xbins_edge = [comp_unq_ - 1.0 / 32.0 for comp_unq_ in comp_unq]
    xbins_edge.append(1 + 1.0 / 32.0)
    print(xbins_edge)
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=composition_list, y=formation_enthalpy_list, bins=[xbins_edge, nbin]
    )
    print(xbins_edge)

    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)

    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (composition_list, formation_enthalpy_list),
        method="linear",
        fill_value=0,
    )  # np.nan)
    #################################################
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(
        composition_list, formation_enthalpy_list, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("Fe concentration")
    plt.ylabel("Mixing enthalpy (Rydberg)")
    plt.title("FePt")
    fig.subplots_adjust(
        left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.1, hspace=0.06
    )
    plt.savefig("formation_enthalpy_colored.png", dpi=400)
    plt.savefig("formation_enthalpy_colored.pdf")


def compute_formation_enthalpy(path_to_filename, elements_list, pure_elements_energy):

    element_counter = dict()

    for atom_type in elements_list:
        element_counter[atom_type] = 0

    df = pandas.read_csv(path_to_filename, header=None, nrows=1)
    energies = np.asarray([float(s) for s in df[0][0].split()])
    total_energy = energies[0]

    df = pandas.read_csv(path_to_filename, header=None, skiprows=1)
    num_atoms = df.shape[0]

    for atom_index in range(0, num_atoms):
        row = df[0][atom_index].split()
        atom_type_str = row[0]
        element_counter[atom_type_str] += 1

    # count the occurrence of the first atom type
    element1 = elements_list[0]
    element2 = elements_list[1]
    composition_element1 = float(element_counter[element1]) / num_atoms
    # print("composition: ", composition_element1)

    # linear_minxing_energy = energy_elemet1 + (energy_element2 - energy_element1) * (1-element1)
    linear_mixing_energy = (
        pure_elements_energy[element1]
        + (pure_elements_energy[element2] - pure_elements_energy[element1])
        * (1 - composition_element1)
    ) * num_atoms

    formation_enthalpy = total_energy - linear_mixing_energy

    return composition_element1, total_energy, linear_mixing_energy, formation_enthalpy


if __name__ == "__main__":
    plot_only()
    # from_energy_to_formation_gibbs(
    #    "./FePt/", elements_list=["26", "78"], temperature_kelvin=0
    # )
