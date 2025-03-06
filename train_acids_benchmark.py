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

#-------------------------------------------
# Define job parameters

parser = argparse.ArgumentParser()

parser.add_argument("output_PATH", type=str)
parser.add_argument("output_file", type=str)

parser.add_argument("property_type", type=str) # 'bond', 'atom', 'mol'
parser.add_argument("property_name", type=str) # see below (property_name_dict)
parser.add_argument("property_aggregation", type=str) # 'none', 'min', 'max', 'boltz', 'min_E'
parser.add_argument("conformer_type", type=str) # 'min_E_DFT', 'random_DFT', 'random_rdkit'
parser.add_argument("N_conformers", type=int) # 1, 5, 8, 1000
parser.add_argument("keep_explicit_hydrogens", type = int) # 0 or 1 or 2 (to keep only functional Hs)

parser.add_argument("model_type", type=str) # 'DimeNetPlusPlus', 'SchNet', 'EGNN', 'PAINN', 'EquiformerV2', 'GIN'
parser.add_argument("ablate_3D", type = int) # 0 or 1
parser.add_argument("pretrained_path", type=str) # path to pretrained model, or str 'none'

parser.add_argument("batch_size", type = int) # 128
parser.add_argument("lr", type = float) # 0.0001
parser.add_argument("seed", type = int)  # 0


parser.add_argument("ensemble_pooling", type = int) # 0 or 1
parser.add_argument("sample_extra_conformers_N", type = int) # 0 or int > 0
parser.add_argument("sample_extra_conformers_type", type = str) # 'rdkit' or 'DFT'
parser.add_argument("active_classification_loss", type = int) # 0 or 1

parser.add_argument("perturb_DFT_conformer", type = str) # 'none' or 'dft_to_rdkit' or 'dft_to_xtb' ##!!!## have to add to submissions.py

parser.add_argument("epoch", type = int)

parser.add_argument("use_atom_features", type = int)

parser.add_argument("train_fraction", type = float)

args = parser.parse_args()

#-------------------------------------------
# Define variables

output_PATH = args.output_PATH + '/' # e.g., 'jobs/'
output_file = args.output_file + '/' # e.g., 'job_0_bond_sterimolL/'
output_dir = output_PATH + output_file

property_type = args.property_type # 'bond', 'atom', or 'mol'
property_name_dict = {
    
    'IR_freq': 'IR_freq', 
    'Sterimol_L': 'Sterimol_L(Å)_morfeus', 
    'Sterimol_B5': 'Sterimol_B5(Å)_morfeus', 
    'Sterimol_B1': 'Sterimol_B1(Å)_morfeus',
    
    'HOMO': 'HOMO', 
    'LUMO': 'LUMO', 
    'polar_aniso': 'polar_aniso(Debye)', 
    'polar_iso': 'polar_iso(Debye)', 
    'dipole': 'dipole(Debye)', 
    'SASA_surface_area': 'SASA_surface_area(Å²)', 
    'SASA_volume': 'SASA_volume(Å³)',
    
    'O3_NBO_charge': 'O3_NBO_charge',
    'C1_NBO_charge': 'C1_NBO_charge',
    'C4_NBO_charge': 'C4_NBO_charge',
    'O2_NBO_charge': 'O2_NBO_charge',
    'H5_NBO_charge': 'H5_NBO_charge',
    'C1_NMR_shift': 'C1_NMR_shift',
    'C4_NMR_shift': 'C4_NMR_shift',
    'H5_NMR_shift': 'H5_NMR_shift',
    'C1_Vbur': 'C1_Vbur',
    'C4_Vbur': 'C4_Vbur',
}
property_name = property_name_dict[args.property_name]
property_aggregation = args.property_aggregation


# no longer used
use_xtb_massive_ensembles = False


conformer_type = args.conformer_type
N_conformers = args.N_conformers
keep_explicit_hydrogens = bool(args.keep_explicit_hydrogens) # False or True
remove_Hs_except_functional = args.keep_explicit_hydrogens == 2 # assumes keep_explicit_hydrogens == True

model_type = args.model_type

pretrained_model = '' 
restart_from_checkpoint = args.pretrained_path == 'checkpoint'
if restart_from_checkpoint:
    pretrained_model = f'{output_dir}/saved_models/model_checkpoint.pt'  
elif args.pretrained_path != 'none':
    if args.pretrained_path[0:12] == 'saved_models':
        pretrained_model = f'{output_dir}{args.pretrained_path}' # path to pre-trained model in local directory
    else:
        pretrained_model = f'{args.pretrained_path}' # full path to saved model (e.g., for transfer learning)

ablate_3D = bool(args.ablate_3D) # False or True


ensemble_pooling = bool(args.ensemble_pooling) #False # True
sample_extra_conformers = (args.sample_extra_conformers_N, args.sample_extra_conformers_type) #(3, 'DFT'), (3, 'rdkit') # how many, and what kind of, "decoy" conformers to sample
if sample_extra_conformers[0] == 0:
    sample_extra_conformers = None
active_classification_loss = bool(args.active_classification_loss) # False # True # whether or not to include active-classification as a training loss (in addition to MSE loss)

ensemble_pooling_layers = [None, 'global_attention', 'self_attention', 'message_passing']
ensemble_pool_layer_type = ensemble_pooling_layers[args.ensemble_pooling] # 'global_attention' or 'self_attention' or 'message_passing'


perturb_DFT_conformer = args.perturb_DFT_conformer 
if perturb_DFT_conformer in ['none', 'None', None]:
    perturb_DFT_conformer = None


batch_size = args.batch_size 
lr = args.lr 

if ensemble_pooling:
    num_workers = 6
else:
    num_workers = 3
N_epochs = 3000
save_every_epoch = 100

seed = args.seed # 0 

use_atom_features = bool(args.use_atom_features) # default should be False

train_fraction = args.train_fraction

# -------------------------------------------
# setting up logging

if not os.path.exists(output_PATH):
    os.makedirs(output_PATH)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_dir + 'saved_models'):
    os.makedirs(output_dir + 'saved_models')

def logger(text, file = output_dir + 'log.txt'):
    with open(file, 'a') as f:
        f.write(text + '\n')

logger(f'job_id: {int(os.environ.get("SLURM_JOB_ID"))}')
logger(f'python command: {sys.argv}')
logger(f'args: {args}')
logger(f' ')


logger(f'property_type: {property_type}')
logger(f'property_name: {property_name}')
logger(f'property_aggregation: {property_aggregation}')
logger(f'conformer_type: {conformer_type}')
logger(f'N_conformers: {N_conformers}')
logger(f'boltzmann_aggregation: {"boltz_weights_G_T_spc"}')
logger(f'T: {298.15}')
logger(f'keep_explicit_hydrogens: {keep_explicit_hydrogens}')
logger(f'remove_Hs_except_functional: {remove_Hs_except_functional}')


logger(f'model_type: {model_type}')
logger(f'pretrained_model: {pretrained_model}')
logger(f'ablate_3D: {ablate_3D}')
logger(f'use_atom_features: {use_atom_features}')

logger(f'num_workers: {num_workers}')
logger(f'batch_size: {batch_size}')
logger(f'lr: {lr}')
logger(f'N_epochs: {N_epochs}')
logger(f'save_every_epoch: {save_every_epoch}')
logger(f'seed: {seed}')
logger(f'starting epoch: {args.epoch}')

logger(f'ensemble_pooling: {ensemble_pooling}')
logger(f'sample_extra_conformers: {sample_extra_conformers}')
logger(f'active_classification_loss: {active_classification_loss}')
logger(f'ensemble_pool_layer_type: {ensemble_pool_layer_type}')

logger(f'perturb_DFT_conformer: {perturb_DFT_conformer}')

logger(f'use_xtb_massive_ensembles: {use_xtb_massive_ensembles}')


# -------------------------------------------
# creating dataframes and splits

if restart_from_checkpoint:
    logger('loading existing dataframes...')
    train_dataframe = pd.read_pickle(f'{output_dir}/train_df.pkl')
    val_dataframe = pd.read_pickle(f'{output_dir}/val_df.pkl')
    test_dataframe = pd.read_pickle(f'{output_dir}/test_df.pkl')
else:
    from process_acid_dataframe import process_acid_dataframe
    
    modeling_dataset = process_acid_dataframe(
        property_name = property_name,
        property_type = property_type,
        property_aggregation = property_aggregation,
        conformer_type = conformer_type,
        N_conformers = N_conformers,
        ensemble_pooling = ensemble_pooling,
        perturb_DFT_conformer = perturb_DFT_conformer,
        sample_extra_conformers = sample_extra_conformers,
        use_xtb_massive_ensembles = use_xtb_massive_ensembles,
        
        remove_clustered_ensembles = True,
        remove_high_energy_ensembles = True,
        
        keep_explicit_hydrogens = keep_explicit_hydrogens, # used for atom mapping only
        use_COOH_search = True,
    )
    
    modeling_dataset = modeling_dataset.sort_values(by = 'Name_int').reset_index(drop = True)
    split_IDs = sorted(list(set(modeling_dataset.Name_int)))
    
    
    """
    # how splitting was done
    # shared_names is set of all Name_ints that can be processed by each model and conformer type
    import random
    splits_per_seed = {}
    
    for seed in [0,1,2,3,4]:
        names_sorted = sorted(list(shared_names))
        
        random.seed(seed)
        random.shuffle(names_sorted)
        
        test_names = names_sorted[0:1000]
        val_names = names_sorted[1000:1500]
        train_names = names_sorted[1500:]
        
        splits_per_seed[f'seed{seed}'] = {
            'train': train_names,
            'val': val_names,
            'test': test_names,
        }
    """
    import pickle
    with open('splits_per_seed.pickle', 'rb') as handle:
        splits_per_seed = pickle.load(handle)
    random.seed(seed)
    test_IDs = set(splits_per_seed[f'seed{seed}']['test'])
    val_IDs = set(splits_per_seed[f'seed{seed}']['val'])
    train_IDs = set(splits_per_seed[f'seed{seed}']['train'])
    train_IDs = list(train_IDs)
    random.shuffle(train_IDs)
    train_IDs = set(train_IDs[0:int(len(train_IDs)*train_fraction)])
    
    
    train_dataframe = modeling_dataset[modeling_dataset.Name_int.isin(train_IDs)].reset_index(drop = True)
    val_dataframe = modeling_dataset[modeling_dataset.Name_int.isin(val_IDs)].reset_index(drop = True)
    test_dataframe = modeling_dataset[modeling_dataset.Name_int.isin(test_IDs)].reset_index(drop = True)
    
    logger('saving dataframes...')
    train_dataframe.to_pickle(output_dir + 'train_df.pkl')
    val_dataframe.to_pickle(output_dir + 'val_df.pkl')
    test_dataframe.to_pickle(output_dir + 'test_df.pkl')


# -------------------------------------------
# initializing datasets and dataloaders

random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

if property_type == 'atom':
    atom_ID_train = np.array(list(train_dataframe.atom_index), dtype = int)
    atom_ID_val = np.array(list(val_dataframe.atom_index), dtype = int)
else:
    atom_ID_train = None
    atom_ID_val = None

if property_type == 'bond':
    bond_ID_1_train = np.array(list(train_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_train = np.array(list(train_dataframe.bond_atom_tuple), dtype = int)[:, 1]
    bond_ID_1_val = np.array(list(val_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_val = np.array(list(val_dataframe.bond_atom_tuple), dtype = int)[:, 1]
else:
    bond_ID_1_train = None
    bond_ID_2_train = None
    bond_ID_1_val = None
    bond_ID_2_val = None

if keep_explicit_hydrogens:
    mols_train = list(train_dataframe.mols)
    mols_val = list(val_dataframe.mols)
else:
    mols_train = list(train_dataframe.mols_noHs)
    mols_val = list(val_dataframe.mols_noHs)


from datasets.dataset_schnet_dimenet import *
train_dataset = Dataset_SchNet_DimeNet(
    property_type = property_type,
    mols = mols_train, 
    targets = list(train_dataframe['y']),
    ligand_ID = np.array(train_dataframe['Name_int']),
    active_conformer = np.array(train_dataframe['active_conformer']) if 'active_conformer' in train_dataframe.columns else None,
    atom_ID = atom_ID_train,
    bond_ID_1 = bond_ID_1_train,
    bond_ID_2 = bond_ID_2_train,
    remove_Hs_except_functional = remove_Hs_except_functional,
)
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


if ensemble_pooling:
    train_loader = torch_geometric.loader.DataLoader(
        dataset = train_dataset,
        num_workers = num_workers,
        batch_sampler = Conformer_Batch_Sampler(train_dataframe, batch_size = batch_size, shuffle = True),
    )
    val_batch_size = 16 if (model_type == 'EquiformerV2') else 100
    val_loader = torch_geometric.loader.DataLoader(
        dataset = val_dataset,
        num_workers = num_workers,
        batch_sampler = Conformer_Batch_Sampler(val_dataframe, batch_size = val_batch_size, shuffle = False),
    )

else:
    train_loader = torch_geometric.loader.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
    )
    val_loader = torch_geometric.loader.DataLoader(
        dataset = val_dataset,
        batch_size = 100,
        shuffle = False,
        num_workers = num_workers,
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
        device = 'cpu',
    )
else:
    raise Exception(f'model type <{model_type}> not implemented')


if pretrained_model != '':
    model.load_state_dict(torch.load(pretrained_model, map_location=next(model.parameters()).device), strict = True)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)


# -------------------------------------------
# training loop

def loop(model, data, training = True, property_type = 'bond', ensemble_pooling = False, active_classification_loss = False, ensembles_contain_actives = False):
    
    if training:
        optimizer.zero_grad()
    
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

    active_class_loss = torch.tensor([0]) # default until overridden
    
    if not ensemble_pooling:
        targets = data.targets
        pred_targets = out[0].squeeze()
        mse_loss = torch.mean(torch.square(targets - pred_targets))
        backprop_loss = mse_loss
        mae = torch.mean(torch.abs(targets - pred_targets))    
        
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
        
        pred_targets = out[0].squeeze()
        mse_loss = torch.mean(torch.square(targets - pred_targets))
        backprop_loss = mse_loss
        mae = torch.mean(torch.abs(targets - pred_targets))      
        
        att_scores = out[2]
        
        if ensembles_contain_actives:
            # compute active/decoy classification loss
            assert att_scores is not None
            
            active_att_scores = att_scores[(data.active_conformer).to(torch.bool)] # pred. prob. of the active conf being active
            active_class_loss = torch.mean(- torch.log(active_att_scores + 1e-8))
            
            if active_classification_loss:
                backprop_loss = backprop_loss + active_class_loss
    
    if training:
        backprop_loss.backward()
        optimizer.step()
    
    return batch_size, backprop_loss.item(), active_class_loss.item(), mae.item(), targets.detach().cpu().numpy(), pred_targets.detach().cpu().numpy()


# -------------------------------------------
# training 

logger('starting to train')

epoch = args.epoch # 0

if restart_from_checkpoint:
    try:
        epoch_train_losses = list(np.load(output_dir + 'training_losses.npy'))
        epoch_val_losses = list(np.load(output_dir + 'validation_losses.npy'))
        epoch_val_active_class_losses = list(np.load(output_dir + 'validation_active_class_losses.npy'))
        epoch_val_maes = list(np.load(output_dir + 'validation_mae.npy'))
        epoch_val_R2s = list(np.load(output_dir + 'validation_R2.npy'))
    except:
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_val_active_class_losses = []
        epoch_val_maes = []
        epoch_val_R2s = []
else:
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_active_class_losses = []
    epoch_val_maes = []
    epoch_val_R2s = []

best_mae = np.inf if (len(epoch_val_maes) == 0) else np.min(np.array(epoch_val_maes))

while epoch < N_epochs:
    epoch += 1
    
    model.train()
    batch_losses = []
    batch_sizes = []
    for batch in train_loader:
        batch_size, loss, active_class_loss, _, _, _ = loop(
            model, 
            batch,
            training = True,
            property_type = property_type,
            ensemble_pooling = ensemble_pooling,
            active_classification_loss = active_classification_loss, 
            ensembles_contain_actives = sample_extra_conformers is not None,
        )
        batch_losses.append(loss)
        batch_sizes.append(batch_size)
        
    epoch_train_losses.append(np.sum(np.array(batch_losses) * np.array(batch_sizes)) / np.sum(np.array(batch_sizes)))
    
    
    model.eval()
    batch_losses = []
    batch_active_class_losses = []
    batch_maes = []
    batch_sizes = []
    val_targets = []
    val_pred_targets = []
    for batch in val_loader:
        with torch.no_grad():
            batch_size, loss, active_class_loss,  mae, v_target, v_pred_target = loop(
                model, 
                batch, 
                training = False, 
                property_type = property_type, 
                ensemble_pooling = ensemble_pooling, 
                active_classification_loss = active_classification_loss,
                ensembles_contain_actives = sample_extra_conformers is not None,
            )
        batch_losses.append(loss)
        batch_active_class_losses.append(active_class_loss)
        batch_maes.append(mae)
        batch_sizes.append(batch_size)
        val_targets.append(v_target)
        val_pred_targets.append(v_pred_target)
    
    epoch_val_losses.append(np.sum(np.array(batch_losses) * np.array(batch_sizes)) / np.sum(np.array(batch_sizes)))
    epoch_val_active_class_losses.append(np.sum(np.array(batch_active_class_losses) * np.array(batch_sizes)) / np.sum(np.array(batch_sizes)))
    epoch_val_maes.append(np.sum(np.array(batch_maes) * np.array(batch_sizes)) / np.sum(np.array(batch_sizes)))
    
    val_targets = np.concatenate(val_targets)
    val_pred_targets = np.concatenate(val_pred_targets)
    val_R2 = np.corrcoef(val_targets, val_pred_targets)[0][1] ** 2
    epoch_val_R2s.append(val_R2)
    
    logger(f'epoch: {epoch}, train_loss: {epoch_train_losses[-1]}, val_loss: {epoch_val_losses[-1]}, val_active_class_loss: {epoch_val_active_class_losses[-1]}, val_mea: {epoch_val_maes[-1]}, val_R2: {epoch_val_R2s[-1]}')
    
    if epoch_val_maes[-1] < best_mae:
        best_mae = epoch_val_maes[-1]
        logger(f'saving best model after epoch {epoch}. Best validation mae: {best_mae}')
        torch.save(model.state_dict(), output_dir + f'saved_models/model_best.pt')
        np.save(output_dir + 'training_losses.npy', np.array(epoch_train_losses))
        np.save(output_dir + 'validation_losses.npy', np.array(epoch_val_losses))
        np.save(output_dir + 'validation_active_class_losses.npy', np.array(epoch_val_active_class_losses))
        np.save(output_dir + 'validation_mae.npy', np.array(epoch_val_maes))
        np.save(output_dir + 'validation_R2.npy', np.array(epoch_val_R2s))
    
    if epoch % 5 == 0:
        #logger(f'saving metrics at epoch {epoch}')
        #torch.save(model.state_dict(), output_dir + f'saved_models/model_checkpoint.pt')
        np.save(output_dir + 'training_losses.npy', np.array(epoch_train_losses))
        np.save(output_dir + 'validation_losses.npy', np.array(epoch_val_losses))
        np.save(output_dir + 'validation_active_class_losses.npy', np.array(epoch_val_active_class_losses))
        np.save(output_dir + 'validation_mae.npy', np.array(epoch_val_maes))
        np.save(output_dir + 'validation_R2.npy', np.array(epoch_val_R2s))

