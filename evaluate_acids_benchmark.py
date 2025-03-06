import torch
import torch_geometric
from datasets.samplers import Conformer_Batch_Sampler

import pandas as pd
import numpy as np
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from rdkit import Chem
from tqdm import tqdm
from copy import deepcopy
import random
import re
import os
import shutil
import argparse
import sys

# -------------------------------------------
# Define job parameters

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=str)
parser.add_argument("property_type", type=str) # 'bond', 'atom', 'mol'
parser.add_argument("model_type", type=str) # 'DimeNetPlusPlus', 'SchNet', 'EGNN', 'PAINN', 'EquiformerV2', 'GIN'
parser.add_argument("keep_explicit_hydrogens", type = int) # 0 or 1 or 2 (to keep only functional Hs)
parser.add_argument("ablate_3D", type = int) # 0 or 1
parser.add_argument("use_atom_features", type = int)
parser.add_argument("ensemble_pooling", type = int) # 0 or 1
parser.add_argument("ensembles_contain_actives", type = int)

parser.add_argument("pretrained_path", type=str) # e.g., '{output_dir}/saved_models/model_best.pt'

args = parser.parse_args()

output_dir = args.output_dir + '/'
property_type = args.property_type
model_type = args.model_type 

keep_explicit_hydrogens = bool(args.keep_explicit_hydrogens) # False or True
remove_Hs_except_functional = args.keep_explicit_hydrogens == 2 # assumes keep_explicit_hydrogens == True

ablate_3D = bool(args.ablate_3D) # False or True
use_atom_features = bool(args.use_atom_features) # default should be True

ensemble_pooling = bool(args.ensemble_pooling) # args.ensemble_pooling in [0,1,2,3]
ensemble_pooling_layers = [None, 'global_attention', 'self_attention', 'message_passing']
ensemble_pool_layer_type = ensemble_pooling_layers[args.ensemble_pooling] # 'global_attention' or 'self_attention' or 'message_passing'

pretrained_model = f'{output_dir}{args.pretrained_path}' # path to pre-trained model in local directory

ensembles_contain_actives = bool(args.ensembles_contain_actives)

# -------------------------------------------

# -------------------------------------------
# setting up logging

def logger(text, file = output_dir + '/eval_log.txt'):
    with open(file, 'a') as f:
        f.write(text + '\n')

logger(f'----------------')
logger(f'EVALUATING MODEL')
logger(f'----------------')

# -------------------------------------------
# loading dataframes and splits

val_dataframe = pd.read_pickle(f'{output_dir}/val_df.pkl')
test_dataframe = pd.read_pickle(f'{output_dir}/test_df.pkl')

# -------------------------------------------
# initializing datasets and dataloaders

seed = 0
random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

if property_type == 'atom':
    atom_ID_val = np.array(list(val_dataframe.atom_index), dtype = int)
    atom_ID_test = np.array(list(test_dataframe.atom_index), dtype = int)
else:
    atom_ID_val = None
    atom_ID_test = None

if property_type == 'bond':
    bond_ID_1_val = np.array(list(val_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_val = np.array(list(val_dataframe.bond_atom_tuple), dtype = int)[:, 1]
    bond_ID_1_test = np.array(list(test_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_test = np.array(list(test_dataframe.bond_atom_tuple), dtype = int)[:, 1]
else:
    bond_ID_1_val = None
    bond_ID_2_val = None
    bond_ID_1_test = None
    bond_ID_2_test = None
    
if keep_explicit_hydrogens:
    mols_val = list(val_dataframe.mols)
    mols_test = list(test_dataframe.mols)
else:
    mols_val = list(val_dataframe.mols_noHs)
    mols_test = list(test_dataframe.mols_noHs)


from datasets.dataset_schnet_dimenet import *
val_dataset = Dataset_SchNet_DimeNet(
    property_type = property_type,
    mols = mols_val, 
    targets = list(val_dataframe['y']),
    ligand_ID = np.array(val_dataframe['Name_int']),
    active_conformer = np.array(val_dataframe['active_conformer']) if 'active_conformer' in val_dataframe.columns else None,
    atom_ID = atom_ID_val,
    bond_ID_1 = bond_ID_1_val,
    bond_ID_2 = bond_ID_2_val,
    remove_Hs_except_functional = remove_Hs_except_functional,
)
test_dataset = Dataset_SchNet_DimeNet(
    property_type = property_type,
    mols = mols_test, 
    targets = list(test_dataframe['y']),
    ligand_ID = np.array(test_dataframe['Name_int']),
    active_conformer = np.array(test_dataframe['active_conformer']) if 'active_conformer' in test_dataframe.columns else None,
    atom_ID = atom_ID_test,
    bond_ID_1 = bond_ID_1_test,
    bond_ID_2 = bond_ID_2_test,
    remove_Hs_except_functional = remove_Hs_except_functional,
)


if ensemble_pooling:
    batch_size = 100
    val_loader = torch_geometric.loader.DataLoader(
        dataset = val_dataset,
        num_workers = 4,
        batch_sampler = Conformer_Batch_Sampler(val_dataframe, batch_size = batch_size, shuffle = False),
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset = test_dataset,
        num_workers = 4,
        batch_sampler = Conformer_Batch_Sampler(test_dataframe, batch_size = batch_size, shuffle = False),
    )
else:
    val_loader = torch_geometric.loader.DataLoader(
        dataset = val_dataset,
        batch_size = 100,
        shuffle = False,
        num_workers = 4,
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset = test_dataset,
        batch_size = 100,
        shuffle = False,
        num_workers = 4,
    )


# -------------------------------------------
# initializing model and optimizer

if model_type in ['DimeNetPlusPlus']:
    from models_benchmark.models_3d import GNN3D

    model = GNN3D(
        model_type = model_type,
        property_type = property_type, # atom, bond, mol 
        out_emb_channels = 128,
        hidden_channels = 128,
        out_channels = 1,
        act = 'swish',
        atom_feature_dim = 53,
        use_atom_features = use_atom_features,
        ablate_3D = ablate_3D,
        ensemble_pooling = ensemble_pooling, 
        ensemble_pool_layer_type = ensemble_pool_layer_type, # 'global_attention', 'self_attention', 'message_passing'
        device = device_name, # 'cpu'
    )

else:
    raise Exception(f'model type <{model_type}> not implemented')


if pretrained_model != '':
    model.load_state_dict(torch.load(pretrained_model, map_location=next(model.parameters()).device), strict = True)


# -------------------------------------------
# evaluation loop

def loop(model, data, property_type = 'bond', ensemble_pooling = False, ensembles_contain_actives = False):
    
    data = data.to(device)
    batch_size = max(data.batch).item() + 1

    ligand_batch_IDs = torch.unique_consecutive(data.ligand_ID, return_inverse = True)[1]
    
    if model_type in ['DimeNetPlusPlus']:
        out = model(
            data = data,
            z = data.x.squeeze(), 
            pos = data.pos, 
            batch = data.batch,
            atom_features = data.atom_features if use_atom_features else None,
            select_atom_index = data.atom_ID_index if property_type == 'atom' else None,
            select_bond_start_atom_index = data.bond_start_ID_index if property_type == 'bond' else None,
            select_bond_end_atom_index = data.bond_end_ID_index if property_type == 'bond' else None,
            ensemble_batch = ligand_batch_IDs,
        )
    
    pred_targets = out[0].squeeze()
  
    if not ensemble_pooling:
        targets = data.targets
        att_scores = torch.ones_like(pred_targets)
        active_att_scores = torch.ones_like(pred_targets)
        
    else: # ensemble_pooling == True
        
        if ensembles_contain_actives:
            # sub-selecting to account for conformer ensemble pooling # this only works if we have explicitly labeled only one of the conformers in the ensemble with active_conf == 1. This was designed to work for the case when we're including the active explicitly in the batch. 
            select_target_indices_bool = (data.active_conformer).to(torch.bool)
            targets = data.targets[select_target_indices_bool]
        
        else:
            # If we're just pooling together random conformers (without necessarily including the active, or at least the active that is explicitly labeld as such), then we just use the conformer grouping provided by ligand_batch_IDs
            unique, inverse = torch.unique(ligand_batch_IDs, sorted=True, return_inverse=True)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            select_target_indices = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
            targets = data.targets[select_target_indices]
        
        att_scores = out[2]
        if ensembles_contain_actives:
            active_att_scores = att_scores[(data.active_conformer).to(torch.bool)] # pred. prob. of the active conf being active
        else:
            active_att_scores = torch.ones_like(pred_targets)

    return targets.detach().cpu().numpy(), pred_targets.detach().cpu().numpy(), att_scores.detach().cpu().numpy(), active_att_scores.detach().cpu().numpy()


# -------------------------------------------
# evaluation

logger('starting to evaluate')

model.eval()
v_targets = []
v_pred_targets = []
v_att_scores = []
v_active_att_scores = []
for batch in val_loader:
    with torch.no_grad():
        targets, pred_targets, att_scores, active_att_scores = loop(
            model, 
            batch, 
            property_type = property_type, 
            ensemble_pooling = ensemble_pooling, 
            ensembles_contain_actives = ensembles_contain_actives,
        )
    v_targets.append(targets)
    v_pred_targets.append(pred_targets)
    v_att_scores.append(att_scores)
    v_active_att_scores.append(active_att_scores)

v_targets = np.concatenate(v_targets)
v_pred_targets = np.concatenate(v_pred_targets)
v_att_scores = np.concatenate(v_att_scores)
v_active_att_scores = np.concatenate(v_active_att_scores)

MAE = np.mean(np.abs(v_targets - v_pred_targets))
R2 = np.corrcoef(v_targets, v_pred_targets)[0][1] ** 2
logger(f'VALIDATION: MAE = {MAE.round(5)}, R2 = {R2.round(5)}')


model.eval()
t_targets = []
t_pred_targets = []
t_att_scores = []
t_active_att_scores = []
for batch in tqdm(test_loader):
    with torch.no_grad():
        targets, pred_targets, att_scores, active_att_scores = loop(
            model, 
            batch, 
            property_type = property_type, 
            ensemble_pooling = ensemble_pooling, 
            ensembles_contain_actives = ensembles_contain_actives,
        )

    t_targets.append(targets)
    t_pred_targets.append(pred_targets)
    t_att_scores.append(att_scores)
    t_active_att_scores.append(active_att_scores)

t_targets = np.concatenate(t_targets)
t_pred_targets = np.concatenate(t_pred_targets)
t_att_scores = np.concatenate(t_att_scores)
t_active_att_scores = np.concatenate(t_active_att_scores)

MAE = np.mean(np.abs(t_targets - t_pred_targets))
R2 = np.corrcoef(t_targets, t_pred_targets)[0][1] ** 2
logger(f'TESTING: MAE = {MAE.round(5)}, R2 = {R2.round(5)}')


np.save(f'{output_dir}/val_predictions.npy', np.stack((v_targets, v_pred_targets), axis = 0))
np.save(f'{output_dir}/test_predictions.npy', np.stack((t_targets, t_pred_targets), axis = 0))

np.save(f'{output_dir}/test_att_scores.npy', t_att_scores)
np.save(f'{output_dir}/test_active_att_scores.npy', t_active_att_scores)