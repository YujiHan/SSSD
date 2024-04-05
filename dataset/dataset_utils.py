'''
Description:
    Utility functions for benchmarking.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import numpy as np
import scanpy
import pandas as pd
import natsort
import torch
import torch.distributions as dist

# --------------------------------
# Load scRNA-seq datasets

device = 1


def loadZebrafishData(data_dir, split_type):
    cnt_data = pd.read_csv(
        "{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0
    )
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index, :]
    cell_stage = meta_data["stage.nice"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage),))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    # -----
    cell_set_meta = pd.read_csv(
        "{}/cell_groups_meta.csv".format(data_dir), header=0, index_col=0
    )
    meta_data = pd.concat([meta_data, cell_set_meta.loc[meta_data.index, :]], axis=1)
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadMammalianData(data_dir, split_type):
    cnt_data = pd.read_csv(
        "{}/{}-norm_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0
    )
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.set_index("NAME")
    meta_data = meta_data.loc[cnt_data.index, :]
    cell_stage = meta_data["orig_ident"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage),))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadDrosophilaData(data_dir, split_type):
    cnt_data = pd.read_csv(
        "{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0
    )
    meta_data = pd.read_csv(
        "{}/subsample_meta_data.csv".format(data_dir), header=0, index_col=0
    )
    meta_data = meta_data.loc[cnt_data.index, :]
    cell_stage = meta_data["time"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage),))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadWOTData(data_dir, split_type):
    cnt_data = pd.read_csv(
        "{}/{}-norm_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0
    )
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    cell_idx = np.where(~np.isnan(meta_data["day"].values))[
        0
    ]  # remove cells with nan labels
    cnt_data = cnt_data.iloc[cell_idx, :]
    meta_data = meta_data.loc[cnt_data.index, :]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage),))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadPancreaticData(data_dir, split_type):
    cnt_data = pd.read_csv(
        "{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0
    )
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index, :]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage),))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadEmbryoidData(data_dir, split_type):
    cnt_data = pd.read_csv(
        "{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0
    )
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index, :]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage),))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


# --------------------------------
# Dataset directories

zebrafish_data_dir = "/home/hanyuji/Workbench/VAE_SSSD/scNODE/data/single_cell/experimental/zebrafish_embryonic/new_processed"
embryoid_data_dir = "/home/hanyuji/Workbench/VAE_SSSD/scNODE/data/single_cell/experimental/embryoid_body/processed"


def loadSCData(data_name, split_type):
    '''
    Main function to load scRNA-seq dataset and pre-process it.
    '''
    print("[ Data={} | Split={} ] Loading data...".format(data_name, split_type))
    if data_name == "zebrafish":
        ann_data = loadZebrafishData(zebrafish_data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types = (
            processed_data.obs["ZF6S-Cluster"]
            .apply(lambda x: "NAN" if pd.isna(x) else x)
            .values
        )
    # elif data_name == "mammalian":
    #     ann_data = loadMammalianData(mammalian_data_dir, split_type)
    #     processed_data = ann_data.copy()
    #     cell_types = processed_data.obs.New_cellType.values
    # elif data_name == "drosophila":
    #     ann_data = loadDrosophilaData(drosophila_data_dir, split_type)
    #     print("Pre-processing...")
    #     ann_data.X = ann_data.X.astype(float)
    #     processed_data = preprocess(ann_data.copy())
    #     cell_types = processed_data.obs.seurat_clusters.values
    # elif data_name == "wot":
    #     ann_data = loadWOTData(wot_data_dir, split_type)
    #     processed_data = ann_data.copy()
    #     cell_types = None
    # elif data_name == "pancreatic":
    #     ann_data = loadPancreaticData(pancreatic_data_dir, split_type)
    #     ann_data.X = ann_data.X.astype(float)
    #     processed_data = preprocess(ann_data.copy())
    #     cell_types = None
    elif data_name == "embryoid":
        ann_data = loadEmbryoidData(embryoid_data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types = None
    else:
        raise ValueError("Unknown data name.")
    cell_tps = ann_data.obs["tp"]
    n_tps = len(np.unique(cell_tps))
    n_genes = ann_data.shape[1]
    return processed_data, cell_tps, cell_types, n_genes, n_tps


# --------------------------------


def traj2Ann(traj_data):
    # traj_data: #trajs, #tps, # features
    traj_data_list = [traj_data[:, t, :] for t in range(traj_data.shape[1])]
    time_step = np.concatenate(
        [np.repeat(t, traj_data.shape[0]) for t in range(traj_data.shape[1])]
    )
    ann_data = scanpy.AnnData(X=np.concatenate(traj_data_list, axis=0))
    ann_data.obs["time_point"] = time_step
    return ann_data


def ann2traj(ann_data):
    time_idx = [
        np.where(ann_data.obs.time_point == t)[0]
        for t in natsort.natsorted(ann_data.obs.time_point.unique())
    ]
    traj_data_list = [ann_data.X[idx, :] for idx in time_idx]
    traj_data = np.asarray(traj_data_list)
    traj_data = np.moveaxis(traj_data, [0, 1, 2], [1, 0, 2])
    return traj_data


# ---------------------------------


def preprocess(ann_data):
    # adopt recipe_zheng17 w/o HVG selection
    # omit scaling part to avoid information leakage
    scanpy.pp.normalize_per_cell(  # normalize with total UMI count per cell
        ann_data, key_n_counts='n_counts_all', counts_per_cell_after=1e4
    )
    scanpy.pp.log1p(ann_data)  # log transform: adata.X = log(adata.X + 1)
    return ann_data


def postprocess(data):
    # data: cell x gene matrix
    if isinstance(data, np.ndarray):
        norm_data = (data / np.sum(data, axis=1)[:, np.newaxis]) * 1e4
        log_data = np.log(norm_data + 1)
    else:
        norm_data = (data / torch.sum(data, dim=1).unsqueeze(dim=1)) * 1e4
        log_data = torch.log(norm_data + 1)
    return log_data
