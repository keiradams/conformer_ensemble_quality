import pandas as pd
import numpy as np
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from tqdm import tqdm
from copy import deepcopy
import random
import re

from process_datasets import create_datasets_acids_massive_ensembles, create_datasets_acids

def get_COOH_idx(mol, has_Hs = True):
    
    if not has_Hs:
        mol_ = rdkit.Chem.AddHs(mol)
    else:
        mol_ = mol
    
    substructure = rdkit.Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    
    indexsall = mol_.GetSubstructMatches(substructure)
    
    if len(indexsall) > 1:
        print('error: multiple matches found')
        return None
    
    o_append=[]
    for i, num in enumerate(range(mol_.GetNumAtoms())):
        if i in indexsall[0]:
            if mol_.GetAtomWithIdx(i).GetSymbol() == 'C':
                C1 = i
            if mol_.GetAtomWithIdx(i).GetSymbol() == 'O':
                o_append.append(i)
    for o in o_append:
        if mol_.GetBondBetweenAtoms(o,C1).GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            O3 = o
        if mol_.GetBondBetweenAtoms(o,C1).GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE:
            O2 = o
    for nei in mol_.GetAtomWithIdx(C1).GetNeighbors():
        if nei.GetSymbol() =='C':
            C4 = nei.GetIdx()
    for nei in mol_.GetAtomWithIdx(O3).GetNeighbors():
        if nei.GetSymbol() =='H':
            H5 = nei.GetIdx()
    
    if not has_Hs:
        H5 = None
    
    return C1, O2, O3, C4, H5


def process_acid_dataframe(
    property_name,
    property_type,
    property_aggregation,
    conformer_type,
    N_conformers,
    ensemble_pooling,
    perturb_DFT_conformer,
    sample_extra_conformers,
    use_xtb_massive_ensembles,
    remove_clustered_ensembles = False,
    remove_high_energy_ensembles = False,
    keep_explicit_hydrogens = False,
    
    use_COOH_search = True,
):
    
    if property_name == 'Sterimol_L(Å)_morfeus':
        query_atoms = ('C1', 'C4')
    if property_name == 'Sterimol_B5(Å)_morfeus':
        query_atoms = ('C1', 'C4')
    if property_name == 'Sterimol_B1(Å)_morfeus':
        query_atoms = ('C1', 'C4')
    if property_name == 'IR_freq':
        query_atoms = ('C1', 'O2')
    if property_name == 'O3_NBO_charge':
        query_atoms = 'O3'
    if property_name == 'C1_NBO_charge':
        query_atoms = 'C1'
    if property_name == 'C4_NBO_charge':
        query_atoms = 'C4'   
    if property_name == 'H5_NBO_charge':
        query_atoms = 'H5'
    if property_name == 'C1_NMR_shift':
        query_atoms = 'C1'
    if property_name == 'C4_NMR_shift':
        query_atoms = 'C4'   
    if property_name == 'H5_NMR_shift':
        query_atoms = 'H5'
    if property_name == 'C1_Vbur':
        query_atoms = 'C1'
    if property_name == 'C4_Vbur':
        query_atoms = 'C4'  


    if use_xtb_massive_ensembles:
        modeling_dataset = create_datasets_acids_massive_ensembles(
            property_aggregation = property_aggregation, 
            property_type = property_type, 
            property_name = property_name, 
            N_conformers = N_conformers, # 1 for active conformer only, or >1 for data augmentation. 
        )
        
    else:
        modeling_dataset = create_datasets_acids(
            property_aggregation = property_aggregation, 
            property_type = property_type, 
            property_name = property_name, 
            conformer_type = conformer_type,
            N_conformers = N_conformers, # 1, unless (property_aggregation == none and conformer_type == random_DFT) --> then set == 1000, or if we are sampling N conformers for data augmentation.
            keep_explicit_hydrogens = False, # ignore; we include molecules with and without Hs here, and select later.
            boltzmann_aggregation = 'boltz_weights_G_T_spc', 
            T = 298.15,
            remove_clustered_ensembles = remove_clustered_ensembles,
            remove_high_energy_ensembles = remove_high_energy_ensembles,
        )
    
    
    if (conformer_type == 'random_rdkit') | (conformer_type == 'random_xtb') | (conformer_type == 'lowest_energy_xtb') | ((conformer_type == 'random_DFT') & (property_aggregation != 'none')):
        # use the following for EITHER:
        # - naive conformer data augmentation with rdkit, xtb, or DFT conformers (N_conformers = 1 (no aug.) or > 1, and ensemble_pooling = False)
        # - naive conformer ensembling with rdkit, xtb, or DFT conformers, without explicitly including DFT actives (N_conformers > 1 and ensemble_pooling = True)
        # - selecting lowest energy xtb conformer (N_conformers = 1)
        
        # Here, each molecule always has N_conformers number of conformers. This is to ensure a balanced dataset in the case of data augmentation.
            # If ensemble_pooling == True, then we explicilty drop repeated conformers.
    
        if conformer_type == 'random_rdkit':
            # replace dft conformers with random rdkit conformers (need to create this pickle file with embeded MMFF molecules)
            # should randomly sample from the list of RDKit conformers (without replacement), only using Name_int (ignoring Conf, as these may all be the same)
            conformer_ensemble = pd.read_pickle('acid_data/rdkit_ensembles_name.pickle')
            conformer_ensemble = {k[0]:conformer_ensemble[k] for k in conformer_ensemble}
        elif conformer_type == 'random_xtb':
            conformer_ensemble = pd.read_pickle('acid_data/rdkit_to_xtb_ensembles_name.pickle')
            conformer_ensemble = {k[0]:conformer_ensemble[k] for k in conformer_ensemble}
        elif conformer_type == 'lowest_energy_xtb':
            assert N_conformers == 1
            conformer_ensemble = pd.read_pickle('acid_data/rdkit_to_xtb_ensembles_name.pickle')
            conformer_ensemble = {k[0]:[conformer_ensemble[k][0]] for k in conformer_ensemble}
        elif (conformer_type == 'random_DFT') & (property_aggregation != 'none'):
            conformer_ensemble = pd.read_pickle('acid_data/dft_to_dft_mols.pickle') 
            conformer_ensemble = {k[0]:conformer_ensemble[k] for k in conformer_ensemble}
        
        sampled_mols = []
        sampled_mols_noHs = []
        atom_index = []
        bond_atom_tuple = []
        
        sampled_confs_from_ensemble = {name: [] for name in set(modeling_dataset.Name_int)}
        
        drop_rows = []
        for i in tqdm(range(len(modeling_dataset))):
            name = modeling_dataset.iloc[i].Name_int
            mol = modeling_dataset.iloc[i].mols
            mol_noHs = modeling_dataset.iloc[i].mols_noHs
            
            if name not in conformer_ensemble:
                print(f'{name} not in conformer ensemble')
                drop_rows.append(i)
                continue
            if len(conformer_ensemble[name]) == 0:
                print(f'{name} not in conformer ensemble') 
                drop_rows.append(i)
                continue
            
            ensemble = conformer_ensemble[name]
            sampled_conf = random.choice( list(set(range(len(ensemble))) - set(sampled_confs_from_ensemble[name])) )
            sampled_mol = ensemble[sampled_conf]
            sampled_mol_noHs = rdkit.Chem.RemoveHs(sampled_mol)
            
            if use_COOH_search:
                if not keep_explicit_hydrogens:
                    C1, O2, O3, C4, H5 = get_COOH_idx(sampled_mol_noHs, has_Hs = False)
                else:
                    C1, O2, O3, C4, H5 = get_COOH_idx(sampled_mol)
                
                if 'atom_index' in modeling_dataset.columns:
                    if query_atoms == 'C1':
                        new_atom_index = C1
                    if query_atoms == 'O2':
                        new_atom_index = O2
                    if query_atoms == 'O3':
                        new_atom_index = O3
                    if query_atoms == 'C4':
                        new_atom_index = C4
                    if query_atoms == 'H5':
                        assert H5 is not None
                        new_atom_index = H5
                    
                    atom_index.append(new_atom_index)
                        
                if 'bond_atom_tuple' in modeling_dataset.columns:
                    if query_atoms == ('C1', 'C4'):
                        new_bond_tuple = (C1, C4)
                    if query_atoms == ('C1', 'O2'):
                        new_bond_tuple = (C1, O2)
                    
                    bond_atom_tuple.append(new_bond_tuple)
            
            else:
                    
                if not keep_explicit_hydrogens:
                    atom_mapping = sampled_mol_noHs.GetSubstructMatch(mol_noHs) # map from old to new
                else:
                    atom_mapping = sampled_mol.GetSubstructMatch(mol) # map from old to new
                
                if (len(atom_mapping) == 0):
                    # this fails for some radical species? Is it okay to just drop the radical species? Should we drop the entire molecule?
                    drop_rows.append(i)
                    print(f'warning: atom map failed for Name_int {name}')
                    continue
                if (max(atom_mapping)+1 != len(atom_mapping)):
                    drop_rows.append(i)
                    print(f'warning: atom map failed for Name_int {name}')
                    continue
                
                if 'atom_index' in modeling_dataset.columns:
                    old_atom_index = modeling_dataset.iloc[i].atom_index
                    new_atom_index = atom_mapping[int(old_atom_index)]
                    atom_index.append(new_atom_index)
                if 'bond_atom_tuple' in modeling_dataset.columns:
                    old_bond_tuple = modeling_dataset.iloc[i].bond_atom_tuple
                    new_bond_tuple = (atom_mapping[old_bond_tuple[0]], atom_mapping[old_bond_tuple[1]])
                    bond_atom_tuple.append(new_bond_tuple)
                
            
            sampled_mols.append(sampled_mol)
            sampled_mols_noHs.append(sampled_mol_noHs)
                
            sampled_confs_from_ensemble[name].append(sampled_conf)
            if len(sampled_confs_from_ensemble[name]) == len(ensemble):
                sampled_confs_from_ensemble[name] = []
        
        
        modeling_dataset.drop(modeling_dataset.index[drop_rows], inplace=True)
        modeling_dataset = modeling_dataset.reset_index(drop = True)
        modeling_dataset['mols'] = sampled_mols
        modeling_dataset['mols_noHs'] = sampled_mols_noHs
        if 'atom_index' in modeling_dataset.columns:
            modeling_dataset['atom_index'] = atom_index
        if 'bond_atom_tuple' in modeling_dataset.columns:
            modeling_dataset['bond_atom_tuple'] = bond_atom_tuple
            
        modeling_dataset.drop(columns = ['Conf'], inplace = True) # in the case of random_DFT, we can actually write some code to preserve the Conf labels for the re-sampled DFT conformers (obtained with conformer_ensemble_df). But, this code is not written. Hence, if we sample random_DFT with property_aggregation != 'none', we lose access to the Conf labeling. This prevents us from doing DFT conformer perturbation when (conformer_type == random_DFT and property_aggregation != 'none').
        
        
        # if ensemble_pooling (for naive ensemble_pooling), drop duplicated conformers (so that we encode just 1 copy of each conformer in the ensemble).
        if ensemble_pooling:
            groups = modeling_dataset.groupby('Name_int', sort = False)
            keep_index = []
            for g in tqdm(groups):
                mols, mols_idx = g[1].mols_noHs, list(g[1].index)            
                coords = []
                keep_mols = []
                for m_idx, m in zip(mols_idx, mols):
                    m_coord = np.array(m.GetConformer().GetPositions())
                    for coord in coords:
                        if np.isclose(m_coord, coord).all():
                            break
                    else:
                        keep_mols.append(m_idx)
                        coords.append(np.array(m.GetConformer().GetPositions()))
                keep_index += keep_mols
            modeling_dataset = modeling_dataset.iloc[keep_index].reset_index(drop = True)
    
    
    if perturb_DFT_conformer in ['dft_to_rdkit', 'dft_to_xtb']: # this needs to be used in conjunction with conformer_type == 'max_DFT', 'min_DFT', 'min_E_DFT', 'max_E_DFT', since the goal is to REPLACE the active conformer. This shouldn't be used with random_DFT (and will throw an error if attempted) because random_DFT does not keep track of Conf, which is required for conformer replacement. We could always change this by keeping track of the Conf indices for the random_DFT conformers (above code block).
        
        assert conformer_type in ['max_DFT', 'min_DFT', 'min_E_DFT', 'max_E_DFT'] # 'random_DFT'
        
        """
        if conformer_type == 'random_DFT': 
            assert property_aggregation != 'none' # while it should be possible to do this perturbation (perturbing a random DFT conformer when predicting an active property), this isn't something we're interested in. If you attempt this, then the code will throw an error because the Conf labels are dropped when sampling random_DFT ? 
        """
        
        # replace dft conformers with re-optimized conformers
        if perturb_DFT_conformer == 'dft_to_rdkit':
            conformer_ensemble = pd.read_pickle('acid_data/dft_to_rdkit_out/dft_to_rdkit_mols.pickle')
        elif perturb_DFT_conformer == 'dft_to_xtb':
            conformer_ensemble = pd.read_pickle('acid_data/dft_to_xtb_out/dft_to_xtb_mols.pickle')
        
        # use 1-to-1 matching between Name_int and Conf
        keys = list(conformer_ensemble.keys())
        name_key, conf_key = [key[0] for key in keys], [key[1] for key in keys]
        conformer_ensemble_df = pd.DataFrame()
        conformer_ensemble_df['Name_int'] = name_key
        conformer_ensemble_df['Conf'] = conf_key
        conformer_ensemble_df['new_mols'] = [conformer_ensemble[k][0] for k in keys]
        conformer_ensemble_df['new_mols_noHs'] = [rdkit.Chem.RemoveHs(conformer_ensemble[k][0]) for k in keys]
        
        modeling_dataset = modeling_dataset.merge(conformer_ensemble_df, on = ['Name_int', 'Conf']).reset_index(drop = True)
        
        if use_COOH_search:
            atom_index = []
            bond_atom_tuple = []
            
            for new_mol, new_mol_noHs in zip(modeling_dataset['new_mols'], modeling_dataset['new_mols_noHs']):
                if not keep_explicit_hydrogens:
                    C1, O2, O3, C4, H5 = get_COOH_idx(new_mol_noHs, has_Hs = False)
                else:
                    C1, O2, O3, C4, H5 = get_COOH_idx(new_mol)
                
                if 'atom_index' in modeling_dataset.columns:
                    if query_atoms == 'C1':
                        new_atom_index = C1
                    if query_atoms == 'O2':
                        new_atom_index = O2
                    if query_atoms == 'O3':
                        new_atom_index = O3
                    if query_atoms == 'C4':
                        new_atom_index = C4
                    if query_atoms == 'H5':
                        assert H5 is not None
                        new_atom_index = H5
                    
                    atom_index.append(new_atom_index)
                        
                if 'bond_atom_tuple' in modeling_dataset.columns:
                    if query_atoms == ('C1', 'C4'):
                        new_bond_tuple = (C1, C4)
                    if query_atoms == ('C1', 'O2'):
                        new_bond_tuple = (C1, O2)
                    
                    bond_atom_tuple.append(new_bond_tuple)
            
            if 'atom_index' in modeling_dataset.columns:
                modeling_dataset['atom_index'] = atom_index
            if 'bond_atom_tuple' in modeling_dataset.columns:
                modeling_dataset['bond_atom_tuple'] = bond_atom_tuple
        
        else:
            if not keep_explicit_hydrogens:
                atom_mappings = [mol_new.GetSubstructMatch(mol_old) for (mol_old, mol_new) in zip(modeling_dataset['mols_noHs'], modeling_dataset['new_mols_noHs'])]
            else:
                atom_mappings = [mol_new.GetSubstructMatch(mol_old) for (mol_old, mol_new) in zip(modeling_dataset['mols'], modeling_dataset['new_mols'])]
                
            if 'atom_index' in modeling_dataset.columns:
                atom_index = [mapp[int(a)] for mapp,a in zip(atom_mappings, modeling_dataset.atom_index)]
                modeling_dataset['atom_index'] = atom_index
            if 'bond_atom_tuple' in modeling_dataset.columns:
                bond_atom_tuple = [(mapp[tup[0]], mapp[tup[1]]) for mapp,tup in zip(atom_mappings, modeling_dataset.bond_atom_tuple)]
                modeling_dataset['bond_atom_tuple'] = bond_atom_tuple
        
        
        modeling_dataset.drop(columns = ['mols', 'mols_noHs'], inplace = True)
        modeling_dataset['mols'] = modeling_dataset['new_mols']
        modeling_dataset['mols_noHs'] = modeling_dataset['new_mols_noHs']
        modeling_dataset.drop(columns = ['new_mols', 'new_mols_noHs'], inplace = True)
    
    
    
    
    if sample_extra_conformers: 
        # only use for models that sample DECOYS (e.g., active DFT is included) for conformer ensembles.
        # unlike the above data augmentation / naive ensembling approach, here we sample without replacement, and thus can have <= N conformers in each ensemble.
        
        assert conformer_type in ['max_DFT', 'min_DFT', 'max_E_DFT', 'min_E_DFT']
        
        decoy_modeling_dataset = pd.DataFrame(np.repeat(modeling_dataset.values, sample_extra_conformers[0], axis=0), columns = modeling_dataset.columns)
        decoy_modeling_dataset['active_conformer'] = np.zeros(len(decoy_modeling_dataset), dtype = int)
        
        if sample_extra_conformers[1] == 'rdkit':
            # (need to create this pickle file with embeded MMFF molecules, clustered to <20 with butina clustering (take centroids this time, not the lowest energy))
            conformer_ensemble = pd.read_pickle('acid_data/rdkit_ensembles_name.pickle')
            conformer_ensemble = {k[0]:conformer_ensemble[k] for k in conformer_ensemble}
        elif sample_extra_conformers[1] == 'DFT':
            remove_keys = set([(n, c) for n,c in zip(decoy_modeling_dataset.Name_int, decoy_modeling_dataset.Conf)]) # removing actives from decoy set
            conformer_ensemble_name_conf = pd.read_pickle('acid_data/dft_to_dft_mols.pickle')
            conformer_ensemble = {k[0]:[] for k in conformer_ensemble_name_conf}
            for k in remove_keys:
                del conformer_ensemble_name_conf[k]
            for k in conformer_ensemble_name_conf:
                conformer_ensemble[k[0]].append(conformer_ensemble_name_conf[k][0])
        elif sample_extra_conformers[1] == 'xtb':
            conformer_ensemble = pd.read_pickle('acid_data/rdkit_to_xtb_ensembles_name.pickle')
            conformer_ensemble = {k[0]:conformer_ensemble[k] for k in conformer_ensemble}
            
        
        sampled_mols = []
        sampled_mols_noHs = []
        atom_index = []
        bond_atom_tuple = []
        
        sampled_confs_from_ensemble = {name: [] for name in set(decoy_modeling_dataset.Name_int)}
        
        drop_rows = []
        for i in tqdm(range(len(decoy_modeling_dataset))):
            name = decoy_modeling_dataset.iloc[i].Name_int
            mol = decoy_modeling_dataset.iloc[i].mols
            mol_noHs = decoy_modeling_dataset.iloc[i].mols_noHs
            
            if name not in conformer_ensemble:
                drop_rows.append(i)
                print(f'{name} not in conformer ensemble') # Should we also drop the entire molecule, including the active DFT ?
                continue
            if len(conformer_ensemble[name]) == 0:
                drop_rows.append(i)
                print(f'{name} not in conformer ensemble')  # Should we also drop the entire molecule, including the active DFT ?
                continue
                
            
            ensemble = conformer_ensemble[name]
            sample_options = list(set(range(len(ensemble))) - set(sampled_confs_from_ensemble[name]))
            if len(sample_options) == 0:
                drop_rows.append(i)
                continue
            sampled_conf = random.choice(sample_options)
            sampled_mol = ensemble[sampled_conf]
            sampled_mol_noHs = rdkit.Chem.RemoveHs(sampled_mol)
            
            if use_COOH_search:
                if not keep_explicit_hydrogens:
                    C1, O2, O3, C4, H5 = get_COOH_idx(sampled_mol_noHs, has_Hs = False)
                else:
                    C1, O2, O3, C4, H5 = get_COOH_idx(sampled_mol)
                
                if 'atom_index' in modeling_dataset.columns:
                    if query_atoms == 'C1':
                        new_atom_index = C1
                    if query_atoms == 'O2':
                        new_atom_index = O2
                    if query_atoms == 'O3':
                        new_atom_index = O3
                    if query_atoms == 'C4':
                        new_atom_index = C4
                    if query_atoms == 'H5':
                        assert H5 is not None
                        new_atom_index = H5
                    atom_index.append(new_atom_index)
                        
                if 'bond_atom_tuple' in modeling_dataset.columns:
                    if query_atoms == ('C1', 'C4'):
                        new_bond_tuple = (C1, C4)
                    if query_atoms == ('C1', 'O2'):
                        new_bond_tuple = (C1, O2)
                    bond_atom_tuple.append(new_bond_tuple)
            else:
                
                if not keep_explicit_hydrogens:
                    atom_mapping = sampled_mol_noHs.GetSubstructMatch(mol_noHs) # map from old to new
                else:
                    atom_mapping = sampled_mol.GetSubstructMatch(mol) # map from old to new
                    
                if (len(atom_mapping) == 0):
                    # this fails for some radical species? Is it okay to just drop the radical species? Should we drop the entire molecule?
                    drop_rows.append(i)
                    print(f'warning: atom map failed for Name_int {name}')
                    continue
                if (max(atom_mapping)+1 != len(atom_mapping)):
                    drop_rows.append(i)
                    print(f'warning: atom map failed for Name_int {name}')
                    continue
                
                if 'atom_index' in decoy_modeling_dataset.columns:
                    old_atom_index = decoy_modeling_dataset.iloc[i].atom_index
                    new_atom_index = atom_mapping[int(old_atom_index)]
                    atom_index.append(new_atom_index)
                if 'bond_atom_tuple' in decoy_modeling_dataset.columns:
                    old_bond_tuple = decoy_modeling_dataset.iloc[i].bond_atom_tuple
                    new_bond_tuple = (atom_mapping[old_bond_tuple[0]], atom_mapping[old_bond_tuple[1]])
                    bond_atom_tuple.append(new_bond_tuple)
            
            
            sampled_mols.append(sampled_mol)
            sampled_mols_noHs.append(sampled_mol_noHs)
                
            sampled_confs_from_ensemble[name].append(sampled_conf)
        
        decoy_modeling_dataset.drop(decoy_modeling_dataset.index[drop_rows], inplace=True)
        decoy_modeling_dataset = decoy_modeling_dataset.reset_index(drop = True)
        decoy_modeling_dataset['mols'] = sampled_mols
        decoy_modeling_dataset['mols_noHs'] = sampled_mols_noHs
        if 'atom_index' in decoy_modeling_dataset.columns:
            decoy_modeling_dataset['atom_index'] = atom_index
        if 'bond_atom_tuple' in decoy_modeling_dataset.columns:
            decoy_modeling_dataset['bond_atom_tuple'] = bond_atom_tuple
        
        # if we want to have DFT decoys labeled with Conf (e.g., to perturb their geometries), then we will need to change the code above to keep track of the Conf IDs of the DFT decoys.
        decoy_modeling_dataset['Conf'] = [None]*len(decoy_modeling_dataset)
        
        modeling_dataset = pd.concat([modeling_dataset, decoy_modeling_dataset]).sort_values(by = ['Name_int', 'active_conformer']).reset_index(drop = True)
        
    
    
    
    return modeling_dataset