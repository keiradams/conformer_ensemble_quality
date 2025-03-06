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

from multiprocessing import Pool
from functools import partial

def get_COOH_idx(mol):
    substructure = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    indexsall = mol.GetSubstructMatches(substructure)
    o_append=[]
    for i, num in enumerate(range(mol.GetNumAtoms())):
        if i in indexsall[0]:
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C':
                C1 = i
            if mol.GetAtomWithIdx(i).GetSymbol() == 'O':
                o_append.append(i)
    for o in o_append:
        if mol.GetBondBetweenAtoms(o,C1).GetBondType() == Chem.rdchem.BondType.SINGLE:
            O3 = o
        if mol.GetBondBetweenAtoms(o,C1).GetBondType() == Chem.rdchem.BondType.DOUBLE:
            O2 = o
    for nei in mol.GetAtomWithIdx(C1).GetNeighbors():
        if nei.GetSymbol() =='C':
            C4 = nei.GetIdx()
    for nei in mol.GetAtomWithIdx(O3).GetNeighbors():
        if nei.GetSymbol() =='H':
            H5 = nei.GetIdx()
    return C1, O2, O3, C4, H5


def generate_rdkit_conformers_from_3D_template(mol_list, use_MMFF = True):
    # the mols in mol_list are assumed to have hydrogens already, and already have a 3D geometry
    rdkit_mols = []
    embed_fail = 0
    opt_fail = 0
    for m in tqdm(mol_list):
        mol = deepcopy(m)
        success = 1
        for _ in range(5):
            success = rdkit.Chem.AllChem.EmbedMolecule(mol)
            if success == 0:
                break
        else:
            # if we fail to embed rdkit mol, just use the provided conformer (not ideal, but will be rare)
            mol = deepcopy(m)
            embed_fail += 1
            rdkit_mols.append(m)
            continue
        
        if use_MMFF:
            unoptimized_mol = deepcopy(mol)
            try:
                rdkit.Chem.AllChem.MMFFOptimizeMolecule(mol)
                mol.GetConformer()
            except:
                mol = unoptimized_mol
                opt_fail += 1
                rdkit_mols.append(mol)
                continue
        
        rdkit_mols.append(mol)
        
    print('# conformers to generate:', len(mol_list))
    print('# failed to embed:', embed_fail)
    print('# failed to optimize:', opt_fail)
    
    return rdkit_mols



def create_datasets_sec_amines(property_aggregation, property_type, property_name, property_name_dict = {}, prop_atom_type = 'N1', conformer_type = 'random_rdkit'):
    # property_aggregation in ['boltz', 'min', 'max', 'min_E']
    # property_type in ['mol', 'atom']
    # property_name in property_name_dict
    # prop_atom_type in ['N1', 'H4']
        
    if (property_aggregation == 'none') & (conformer_type in ['max_DFT', 'min_DFT', 'min_E_DFT']):
        convert_prop_aggregation = {'max_DFT': 'max', 'min_DFT': 'min', 'min_E_DFT': 'min_E'}
        property_aggregation = convert_prop_aggregation[conformer_type]
    elif property_aggregation == 'none':
        raise Exception(f"{property_aggregation} cannot be 'none' if conformer_type is not one of {['max_DFT', 'min_DFT', 'min_E_DFT']}")
    
    prop = property_name_dict[f'{property_name}_{property_aggregation}']
    
    property_abbrev_dict = {
        'dipole': 'dipole(Debye)', 
        'pyr_agranat': 'pyramidalization_Agranat-Radhakrishnan',
        #'pyr_gavrish': 'pyramidalization_Gavrish', 
        'NBO_LP_energy': 'NBO_LP_energy', 
        'NBO_LP_occupancy': 'NBO_LP_occupancy', 
        'Vbur': '%Vbur_2.5Å',
        
        'NBO_charge_H': 'NBO_charge',
        'NBO_charge_N': 'NBO_charge',
        
        'NMR_shift_H': 'NMR_shift',
        'NMR_shift_N': 'NMR_shift',
        
    }
    
    csv = pd.read_pickle('secondary_amine_data/20230602_sec_amine_mol_data_with_3D_processed.pkl')
    csv = csv.drop_duplicates('Name_int').reset_index(drop = True) # keeping only 1 dummy conformer
    
    if property_type == 'mol':
        mol_df = pd.read_csv('secondary_amine_data/20230823_sec_amine_mol_data_with_Set.csv.gz').reset_index(drop = True)
        mol_df['name'] = mol_df['Compound_Name.1']
        mol_df['Name_int'] = [20000 + int(s[4:]) for s in mol_df['name']]
        mol_df = mol_df.merge(csv[['Name_int', 'H4_canonical', 'N1_canonical', 'smiles', 'mols']], on = 'Name_int')
        y_dataset = mol_df[['Name_int', 'smiles', prop]]
    
    if property_type == 'atom':
        atom_df = pd.read_csv('secondary_amine_data/20230823_sec_amine_atom_data_with_Set.csv.gz').reset_index(drop = True)
        atom_df = atom_df.dropna(thresh = 5).reset_index(drop = True)
        atom_df['name'] = atom_df['Compound_Name.1']
        atom_df['Name_int'] = [20000 + int(s[4:]) for s in atom_df['name']]
        atom_df = atom_df.merge(csv[['Name_int', 'H4_canonical', 'N1_canonical', 'smiles', 'mols']], on = 'Name_int')
        
        y_dataset = atom_df[['Name_int', 'smiles', 'atom_type', 'H4_canonical', 'N1_canonical', prop]]
        y_dataset = y_dataset.dropna(subset = [prop]).reset_index(drop = True)
        
        y_dataset = y_dataset[y_dataset.atom_type == prop_atom_type].reset_index(drop = True)
        if prop_atom_type == 'N1':
            y_dataset['atom_index'] = y_dataset['N1_canonical']
        if prop_atom_type in ['H', 'H4']:
            y_dataset['atom_index'] = y_dataset['H4_canonical']
        y_dataset = y_dataset.drop(columns = ['H4_canonical', 'N1_canonical'])
    
    
    y_dataset['Name_int'] = np.array(y_dataset['Name_int'], dtype = int)
    y_dataset['mols'] = [None]*len(y_dataset)
    y_dataset['mols_noHs'] = [None]*len(y_dataset)
    y_dataset = y_dataset.rename(columns = {prop : 'y'})
    
    # replace the None mol objects in y_dataset with selected "active" DFT mols
    if conformer_type in ['max_DFT', 'min_DFT', 'min_E_DFT']:
        
        assert property_aggregation in ['min_E', 'min', 'max']
        
        if property_type == 'mol':
            mol_csv = pd.read_pickle('secondary_amine_data/20230602_sec_amine_mol_data_with_3D_processed.pkl')
            mol_csv = mol_csv[['Conf', 'Name_int', 'mols', 'G(T)_spc(Hartree)', property_abbrev_dict[property_name]]]
            
            groups = mol_csv.groupby('Name_int')
            keep_indices = []
            for g in groups:
                group_indices = list(g[1].index)
                if property_aggregation == 'min_E':
                    select_index = group_indices[np.argmin(g[1]['G(T)_spc(Hartree)'])]
                if property_aggregation == 'min':
                    select_index = group_indices[np.argmin(g[1][property_abbrev_dict[property_name]])]
                if property_aggregation == 'max':
                    select_index = group_indices[np.argmax(g[1][property_abbrev_dict[property_name]])]
                keep_indices.append(select_index)
            keep_indices = sorted(keep_indices)
            mol_csv = mol_csv.loc[keep_indices].sort_values(by = ['Name_int'])
            
            modeling_dataset = y_dataset.drop(['mols'], axis=1).merge(mol_csv[['Name_int', 'mols']], on = 'Name_int')
            modeling_dataset['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in modeling_dataset.mols]
            y_dataset = modeling_dataset.reset_index(drop = True)
        
        if property_type == 'atom':
            atom_csv = pd.read_pickle('secondary_amine_data/20230907_sec_amine_atom_data_with_3D_processed.pkl')
            atom_csv = atom_csv[['Conf', 'Name_int', 'mols', 'G(T)_spc(Hartree)', 'atom_type', 'H4_canonical', 'N1_canonical', property_abbrev_dict[property_name]]]
            atom_csv = atom_csv[atom_csv.atom_type == prop_atom_type].reset_index(drop = True).dropna(subset = [property_abbrev_dict[property_name]]).reset_index(drop = True)

            groups = atom_csv.groupby('Name_int')
            keep_indices = []
            for g in groups:
                group_indices = list(g[1].index)
                if property_aggregation == 'min_E':
                    select_index = group_indices[np.argmin(g[1]['G(T)_spc(Hartree)'])]
                if property_aggregation == 'min':
                    select_index = group_indices[np.argmin(g[1][property_abbrev_dict[property_name]])]
                if property_aggregation == 'max':
                    select_index = group_indices[np.argmax(g[1][property_abbrev_dict[property_name]])]
                keep_indices.append(select_index)
            keep_indices = sorted(keep_indices)
            atom_csv_ = atom_csv.loc[keep_indices].sort_values(by = ['Name_int'])
            
            if prop_atom_type == 'N1':
                atom_csv_['atom_index'] = atom_csv_['N1_canonical']
            if prop_atom_type in ['H', 'H4']:
                atom_csv_['atom_index'] = atom_csv_['H4_canonical']
            
            modeling_dataset = y_dataset.drop(['mols', 'atom_index'], axis =1).merge(atom_csv_[['Name_int', 'mols', 'atom_index']], on = 'Name_int') #, suffixes = ['_x', None]
            modeling_dataset['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in modeling_dataset.mols]
            y_dataset = modeling_dataset.reset_index(drop = True)
    
    return y_dataset




def create_datasets_amines(property_aggregation, property_type, property_name, property_name_dict = {}, prop_atom_type = 'N1', conformer_type = 'random_rdkit'):
    # property_aggregation in ['boltz', 'min', 'max', 'min_E']
    # property_type in ['mol', 'atom', 'bond']
    # property_name in mol_property_dict, atom_property_dict, or bond_property_dict
    # prop_atom_type in ['N1', 'C2', 'H3']
        
    if (property_aggregation == 'none') & (conformer_type in ['max_DFT', 'min_DFT', 'min_E_DFT']):
        convert_prop_aggregation = {'max_DFT': 'max', 'min_DFT': 'min', 'min_E_DFT': 'min_E'}
        property_aggregation = convert_prop_aggregation[conformer_type]
    elif property_aggregation == 'none':
        raise Exception(f"{property_aggregation} cannot be 'none' if conformer_type is not one of {['max_DFT', 'min_DFT', 'min_E_DFT']}")
    
    prop = property_name_dict[f'{property_name}_{property_aggregation}']
    
    try:
        csv = pd.read_pickle('amine_data/20230411_amine_mol_data_with_3D_processed.pkl')
    except:
        csv = pd.read_pickle('amine_data/20230411_amine_mol_data_with_3D_processed_copy20230516.pkl') # this just uses 'mol_block_from_mols' instead of 'mols' to avoid pickling issues
        csv['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in csv['mol_block_from_mols']]
    csv = csv.drop_duplicates('Name_int').reset_index(drop = True) # keeping only 1 dummy conformer
    
    
    # need to get index -> Name_int mapping from mol_df
    mol_df = pd.read_csv('amine_data/20230228_amine_mol_data_with_Set.csv.gz').reset_index(drop = True)
    #mol_df = mol_df.rename(columns = {'index': 'Name_int'})
    mol_df['Name_int'] = [int(s[1:]) for s in mol_df['Compound_Name']]
    mol_df = mol_df.merge(csv[['Name_int', 'N1_canonical', 'C2_canonical', 'H3_canonical', 'H4_canonical', 'smiles', 'mols']], on = 'Name_int')
    index_to_Name_int_map = {int(i) : n for i,n in zip(mol_df['index'], mol_df['Name_int'])}
    
    if property_type == 'mol':
        y_dataset = mol_df[['Name_int', 'smiles', prop]]
    
    if property_type == 'atom':
        atom_df = pd.read_csv('amine_data/20230228_amine_atom_data.csv.gz')
        atom_df = atom_df.dropna(thresh = 4).reset_index(drop = True)
        #atom_df = atom_df.rename(columns = {'index': 'Name_int'})
        atom_df['Name_int'] = [index_to_Name_int_map[int(i)] if int(i) in index_to_Name_int_map else None for i in atom_df['index'] ]
        atom_df = atom_df.dropna(subset = ['Name_int']).reset_index(drop = True)
        atom_df['Name_int'] = np.array(atom_df['Name_int'], dtype = int)
        
        atom_df = atom_df.merge(csv[['Name_int', 'N1_canonical', 'C2_canonical', 'H3_canonical', 'H4_canonical', 'smiles', 'mols']], on = 'Name_int')
        
        y_dataset = atom_df[['Name_int', 'smiles', 'atom_type', 'N1_canonical', 'C2_canonical', 'H3_canonical', 'H4_canonical', prop]]
        y_dataset = y_dataset.dropna(subset = [prop]).reset_index(drop = True)
        
        y_dataset = y_dataset[y_dataset.atom_type == prop_atom_type].reset_index(drop = True)
        if prop_atom_type == 'N1':
            y_dataset['atom_index'] = y_dataset['N1_canonical']
        if prop_atom_type == 'C2':
            y_dataset['atom_index'] = y_dataset['C2_canonical']
        if prop_atom_type in ['H', 'H3', 'H4']:
            y_dataset['atom_index'] = y_dataset['H3_canonical'] # we only use the H3 atom for predicting H properties
            
        y_dataset = y_dataset.drop(columns = ['N1_canonical', 'C2_canonical', 'H3_canonical', 'H4_canonical'])
        
    if property_type == 'bond':
        bond_df = pd.read_csv('amine_data/20230228_amine_bond_data.csv.gz')
        bond_df = bond_df.dropna(thresh = 4).reset_index(drop = True)
        bond_df['Name_int'] = [index_to_Name_int_map[int(i)] if int(i) in index_to_Name_int_map else None for i in bond_df['index'] ]
        bond_df = bond_df.dropna(subset = ['Name_int']).reset_index(drop = True)
        bond_df['Name_int'] = np.array(bond_df['Name_int'], dtype = int)
        
        #bond_df = bond_df.rename(columns = {'index': 'Name_int'})
        bond_df = bond_df.merge(csv[['Name_int', 'N1_canonical', 'C2_canonical', 'H3_canonical', 'H4_canonical', 'smiles', 'mols']], on = 'Name_int')
        bond_df['bond_atom_tuple'] = [(i,j) for i,j in zip(bond_df.N1_canonical, bond_df.C2_canonical)]
        y_dataset = bond_df[['Name_int', 'smiles', 'bond_atom_tuple', prop]]
    
    y_dataset['Name_int'] = np.array(y_dataset['Name_int'], dtype = int)
    y_dataset['mols'] = [None]*len(y_dataset)
    y_dataset['mols_noHs'] = [None]*len(y_dataset)
    y_dataset = y_dataset.rename(columns = {prop : 'y'})
    
    
    # replace the None mol objects in y_dataset with selected "active" DFT mols
    if conformer_type in ['max_DFT', 'min_DFT', 'min_E_DFT']:
        
        property_abbrev_dict = {
            'dipole': 'dipole(Debye)', 
            'pyr_agranat': 'pyramidalization_Agranat-Radhakrishnan',
            'pyr_gavrish': 'pyramidalization_Gavrish', 
            'NBO_LP_energy': 'NBO_LP_energy', 
            'NBO_LP_occupancy': 'NBO_LP_occupancy', 
            'Vbur': '%Vbur_2.5Å',
            'Sterimol_L': 'Sterimol_L(Å)_morfeus', 
            'Sterimol_B1': 'Sterimol_B1(Å)_morfeus',
            'Sterimol_B5': 'Sterimol_B5(Å)_morfeus',
            
            'NBO_charge_H_min': 'NBO_charge_H_min',
            'NBO_charge_H_avg': 'NBO_charge_H_avg',
            'NMR_shift_H_avg': 'NMR_shift_H_avg',
            
            # 11/22/23
            'NBO_charge_C': 'NBO_charge', # not tested, but should work
            'NBO_charge_N': 'NBO_charge', # not tested, but should work
            'NMR_shift_C': 'NMR_shift', # not tested, but should work
            'NMR_shift_N': 'NMR_shift', # not tested, but should work
        }
        
        
        if property_type == 'mol':
            #mol_csv = pd.read_pickle('amine_data/20230320_amine_mol_data_with_3D_processed.pkl')
            mol_csv = pd.read_pickle('amine_data/20230411_amine_mol_data_with_3D_processed.pkl')
            mol_csv = mol_csv[['Conf', 'Name_int', 'mols', 'G(T)_spc(Hartree)', property_abbrev_dict[property_name]]]
            
            groups = mol_csv.groupby('Name_int')
            keep_indices = []
            for g in groups:
                group_indices = list(g[1].index)
                if property_aggregation == 'min_E':
                    select_index = group_indices[np.argmin(g[1]['G(T)_spc(Hartree)'])]
                if property_aggregation == 'min':
                    select_index = group_indices[np.argmin(g[1][property_abbrev_dict[property_name]])]
                if property_aggregation == 'max':
                    select_index = group_indices[np.argmax(g[1][property_abbrev_dict[property_name]])]
                keep_indices.append(select_index)
            keep_indices = sorted(keep_indices)
            mol_csv = mol_csv.loc[keep_indices].sort_values(by = ['Name_int'])
            
            modeling_dataset = y_dataset.drop(['mols'], axis =1).merge(mol_csv[['Name_int', 'mols']], on = 'Name_int')
            modeling_dataset['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in modeling_dataset.mols]
            y_dataset = modeling_dataset.reset_index(drop = True)
            
        if property_type == 'atom':
            #atom_csv = pd.read_pickle('amine_data/20230320_amine_atom_data_with_3D_processed.pkl')
            atom_csv = pd.read_pickle('amine_data/20230411_amine_atom_data_with_3D_processed.pkl')
            atom_csv = atom_csv[['Conf', 'Name_int', 'mols', 'G(T)_spc(Hartree)', 'atom_type', 'N1_canonical', 'C2_canonical', 'H3_canonical', 'H4_canonical', property_abbrev_dict[property_name]]]
            atom_csv = atom_csv[atom_csv.atom_type == prop_atom_type].reset_index(drop = True).dropna(subset = [property_abbrev_dict[property_name]]).reset_index(drop = True)
            
            groups = atom_csv.groupby('Name_int')
            keep_indices = []
            for g in groups:
                group_indices = list(g[1].index)
                if property_aggregation == 'min_E':
                    select_index = group_indices[np.argmin(g[1]['G(T)_spc(Hartree)'])]
                if property_aggregation == 'min':
                    select_index = group_indices[np.argmin(g[1][property_abbrev_dict[property_name]])]
                if property_aggregation == 'max':
                    select_index = group_indices[np.argmax(g[1][property_abbrev_dict[property_name]])]
                keep_indices.append(select_index)
            keep_indices = sorted(keep_indices)
            atom_csv = atom_csv.loc[keep_indices].sort_values(by = ['Name_int'])
            
            if prop_atom_type == 'N1':
                atom_csv['atom_index'] = atom_csv['N1_canonical']
            if prop_atom_type == 'C2':
                atom_csv['atom_index'] = atom_csv['C2_canonical']
            if prop_atom_type in ['H', 'H3', 'H4']:
                atom_csv['atom_index'] = atom_csv['H3_canonical'] # we only use the H3 atom for predicting H properties
                
            modeling_dataset = y_dataset.drop(['mols', 'atom_index'], axis =1).merge(atom_csv[['Name_int', 'mols', 'atom_index']], on = 'Name_int') #, suffixes = ['_x', None]
            modeling_dataset['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in modeling_dataset.mols]
            y_dataset = modeling_dataset.reset_index(drop = True)
        
        if property_type == 'bond':
            #bond_csv = pd.read_pickle('amine_data/20230320_amine_bond_data_with_3D_processed.pkl')
            bond_csv = pd.read_pickle('amine_data/20230411_amine_bond_data_with_3D_processed.pkl')
            bond_csv = bond_csv[['Conf', 'Name_int', 'mols', 'G(T)_spc(Hartree)', 'bond_atom_tuple', property_abbrev_dict[property_name]]]
            
            groups = bond_csv.groupby('Name_int')
            keep_indices = []
            for g in groups:
                group_indices = list(g[1].index)
                if property_aggregation == 'min_E':
                    select_index = group_indices[np.argmin(g[1]['G(T)_spc(Hartree)'])]
                if property_aggregation == 'min':
                    select_index = group_indices[np.argmin(g[1][property_abbrev_dict[property_name]])]
                if property_aggregation == 'max':
                    select_index = group_indices[np.argmax(g[1][property_abbrev_dict[property_name]])]
                keep_indices.append(select_index)
            keep_indices = sorted(keep_indices)
            bond_csv = bond_csv.loc[keep_indices].sort_values(by = ['Name_int'])
            
            modeling_dataset = y_dataset.drop(['mols', 'bond_atom_tuple'], axis =1).merge(bond_csv[['Name_int', 'mols', 'bond_atom_tuple']], on = 'Name_int')
            modeling_dataset['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in modeling_dataset.mols]
            y_dataset = modeling_dataset.reset_index(drop = True)
    
    return y_dataset



def create_datasets_acids(property_aggregation, property_type, property_name, conformer_type, N_conformers, keep_explicit_hydrogens, boltzmann_aggregation = 'boltz_weights_G_T_spc', T = 298.15, sample_extra_conformers = None, remove_clustered_ensembles = False, remove_high_energy_ensembles = False):
    # property_aggregation: ['none', 'min', 'max', 'boltz', 'min_E', 'max_E']
    # property_type: ['atom', 'bond', 'mol']
    # property_name:
        # atom: 
        # bond:
        # mol:
    # conformer_type: ['min_E_DFT', 'min_E_DFT', 'random_DFT', 'random_rdkit']
    # N_conformers: int (1, 2, ..., 1000 (for all)) # relevant only if conformer_type in ['random_DFT', 'random_rdkit']
    #keep_explicit_hydrogens: bool
    # boltzmann_aggregation: ['boltz_weights_G_T_spc', 'boltz_weights_qh_G_T_spc']
    # T: float # defines temperature for boltzmann averaging
    
    # sample_extra_conformers == None or (N_extra, 'rdkit' or 'DFT')
    
    
    # THERE ARE A FEW (<20) MOLECULES WHOSE RDKIT/XTB CONFORMERS HAVE DIFFERENT STEREOCHEMISTRY -- should we remove these molecules?
    
    # step 1
    mol_dataset = pd.read_csv('acid_data/data/20230112_data/mol_sdf.csv.gz')
    mol_dataset['Name_int'] = [int(n[2:]) for n in list(mol_dataset.Name)]
    
    mol_dataset['is_clustered'] = np.array(['clust' in i for i in list(mol_dataset.ID)])
    groups = mol_dataset.groupby('Name_int')
    clustered_groups = []
    high_energy_groups = []
    for g in groups:
        if remove_clustered_ensembles:
            if np.array(g[1]['is_clustered']).any():
                clustered_groups.append(g[0])
        if remove_high_energy_ensembles:
            G_T_spc = np.array(g[1]['G(T)_spc(Hartree)']) * 627.509
            if max(G_T_spc - min(G_T_spc)) > 5.0:
                high_energy_groups.append(g[0])
    remove_groups = set(clustered_groups + high_energy_groups)  
    print(f'removing {len(remove_groups)} ensembles')
    
    
    suppl = rdkit.Chem.rdmolfiles.ForwardSDMolSupplier('acid_data/data/20230112_data/mol_sdf.sdf', removeHs = False)
    mols = []
    mols_name_int = []
    mols_conf = []
    mols_block = []
    #mols_prop_dict = [] # too slow; mol_dataset already contains this
    for mol in tqdm(suppl):
        mols.append(mol)
        mols_name_int.append(int(mol.GetProp('Name')[2:]))
        mols_conf.append(int(mol.GetProp('Conf')))
        mols_block.append(rdkit.Chem.MolToMolBlock(mol))
        #mols_prop_dict.append(mol.GetPropsAsDict())
    
    assert mols_name_int == list(mol_dataset['Name_int'])
    assert mols_conf == list(mol_dataset['Conf'])
    
    mol_dataset['mol_block'] = mols_block
    
    # there are about 94 duplicated mols in the data
    mol_dataset = mol_dataset.sort_values(by = ['Name_int', 'Conf', 'mol_block']).reset_index(drop = True)
    print(len(mol_dataset))
    mol_dataset = mol_dataset.drop_duplicates(subset = ['mol_block']).reset_index(drop = True)
    print(len(mol_dataset))
    
    # step 2
    kb = 3.166811563e-6 # Hartree / K
    # T = 298.15 # K (default; defined in function initialization)
    
    groups = mol_dataset.groupby('Name_int', sort = False)
    groups_boltz_weights_G_T_spc = np.zeros(len(mol_dataset))
    groups_boltz_weights_qh_G_T_spc = np.zeros(len(mol_dataset))
    groups_min_E_conf = np.zeros(len(mol_dataset), dtype = int)
    groups_max_E_conf = np.zeros(len(mol_dataset), dtype = int)
    pointer = 0
    for g in tqdm(groups):
        name = g[0]
        confs = np.array(g[1]['Conf'])
        G_T_spc = np.array(g[1]['G(T)_spc(Hartree)'])
        qh_G_T_spc = np.array(g[1]['qh_G(T)_spc(Hartree)'])
        
        min_E_conf = np.zeros(len(g[1]), dtype = int)
        min_E_conf[np.argmin(G_T_spc)] = 1
        groups_min_E_conf[pointer:pointer+len(g[1])] = min_E_conf
        
        max_E_conf = np.zeros(len(g[1]), dtype = int)
        max_E_conf[np.argmax(G_T_spc)] = 1
        groups_max_E_conf[pointer:pointer+len(g[1])] = max_E_conf
        
        E = G_T_spc
        E_min = np.min(E)
        #E_max = np.max(E)
        boltz_weights_G_T_spc = np.exp(-((E - E_min)) / (kb*T)) / np.sum(np.exp(-((E - E_min)) / (kb*T)))
        groups_boltz_weights_G_T_spc[pointer:pointer+len(g[1])] = boltz_weights_G_T_spc
    
        E = qh_G_T_spc
        E_min = np.min(E)
        #E_max = np.max(E)
        boltz_weights_qh_G_T_spc = np.exp(-((E - E_min)) / (kb*T)) / np.sum(np.exp(-((E - E_min)) / (kb*T)))
        groups_boltz_weights_qh_G_T_spc[pointer:pointer+len(g[1])] = boltz_weights_qh_G_T_spc
        
        pointer += len(g[1])
    
    mol_dataset['boltz_weights_G_T_spc'] = groups_boltz_weights_G_T_spc
    mol_dataset['boltz_weights_qh_G_T_spc'] = groups_boltz_weights_qh_G_T_spc
    mol_dataset['min_E_conf'] = groups_min_E_conf
    mol_dataset['max_E_conf'] = groups_max_E_conf
    
    
    # step 3
    bond_dataset = pd.read_csv('acid_data/data/20230112_data/Bond_Properties_for_3D_formated_removed.csv.gz')
    
    bond_dataset['Conf'] = [int(re.findall(r'\d+', l)[1]) for l in bond_dataset.log_name]
    bond_dataset['Name_int'] = [int(n[2:]) for n in bond_dataset.name]
    
    bond_dataset = bond_dataset.sort_values(by = ['Name_int', 'Conf', 'mol_block']).reset_index(drop = True)
    print(len(bond_dataset))
    bond_dataset = bond_dataset.drop_duplicates().reset_index(drop = True)
    print(len(bond_dataset))
    
    bond_dataset = bond_dataset.loc[bond_dataset[['IR_freq', 'Sterimol_L(Å)_morfeus', 'Sterimol_B5(Å)_morfeus', 'Sterimol_B1(Å)_morfeus']].dropna(how = 'all').index].reset_index(drop = True)
    print(len(bond_dataset))
    
    
    bond_indices = list(bond_dataset.bond_index)
    bond_dataset_atom_tuple = []
    for i,m in tqdm(enumerate(list(bond_dataset.mol_block)), total = len(bond_dataset)):
        b = rdkit.Chem.MolFromMolBlock(m, removeHs = False).GetBondWithIdx(int(bond_indices[i]))
        bond_tuple = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        bond_dataset_atom_tuple.append(bond_tuple)
    bond_dataset['bond_atom_tuple'] = bond_dataset_atom_tuple
    
    
    # step 4
    atom_dataset = pd.read_csv('acid_data/data/20230112_data/Atom_Properties_for_3D_formated_removed.csv.gz')
    atom_dataset['Conf'] = [int(re.findall(r'\d+', l)[1]) for l in atom_dataset.log_name]
    atom_dataset['Name_int'] = [int(n[2:]) for n in atom_dataset.name]
    atom_dataset = atom_dataset.sort_values(by = ['Name_int', 'Conf', 'mol_block']).reset_index(drop = True)
    print(len(atom_dataset))
    atom_dataset = atom_dataset.drop_duplicates().reset_index(drop = True)
    print(len(atom_dataset))
    
    
    # step 5
    # splitting charge, nmr, and Vbur tasks into single-tasks for each atom-type
    atom_dataset_atom_types = list(atom_dataset.atom_type)
    atom_dataset_NBO_charges = list(atom_dataset.NBO_charge)
    atom_dataset_NMR_shift = list(atom_dataset.NMR_shift)
    atom_dataset_Vbur = list(atom_dataset['%Vbur_3.0Å'])
    
    O3_NBO_charge = [atom_dataset_NBO_charges[i] if atom_dataset_atom_types[i] == 'O3' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    C1_NBO_charge = [atom_dataset_NBO_charges[i] if atom_dataset_atom_types[i] == 'C1' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    C4_NBO_charge = [atom_dataset_NBO_charges[i] if atom_dataset_atom_types[i] == 'C4' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    O2_NBO_charge = [atom_dataset_NBO_charges[i] if atom_dataset_atom_types[i] == 'O2' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    H5_NBO_charge = [atom_dataset_NBO_charges[i] if atom_dataset_atom_types[i] == 'H5' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    
    C1_NMR_shift = [atom_dataset_NMR_shift[i] if atom_dataset_atom_types[i] == 'C1' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    C4_NMR_shift = [atom_dataset_NMR_shift[i] if atom_dataset_atom_types[i] == 'C4' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    H5_NMR_shift = [atom_dataset_NMR_shift[i] if atom_dataset_atom_types[i] == 'H5' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    
    C1_Vbur = [atom_dataset_Vbur[i] if atom_dataset_atom_types[i] == 'C1' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    C4_Vbur = [atom_dataset_Vbur[i] if atom_dataset_atom_types[i] == 'C4' else np.nan for i in range(len(atom_dataset_atom_types)) ]
    
    atom_dataset['O3_NBO_charge'] = O3_NBO_charge
    atom_dataset['C1_NBO_charge'] = C1_NBO_charge
    atom_dataset['C4_NBO_charge'] = C4_NBO_charge
    atom_dataset['O2_NBO_charge'] = O2_NBO_charge
    atom_dataset['H5_NBO_charge'] = H5_NBO_charge
    
    atom_dataset['C1_NMR_shift'] = C1_NMR_shift
    atom_dataset['C4_NMR_shift'] = C4_NMR_shift
    atom_dataset['H5_NMR_shift'] = H5_NMR_shift
    
    atom_dataset['C1_Vbur'] = C1_Vbur
    atom_dataset['C4_Vbur'] = C4_Vbur
    
    
    # step 6
    # QCing atom_dataset and mol_dataset and bond_dataset
    
    # these should all be the same!
    print(sum(~np.isnan(atom_dataset['C1_NMR_shift'])), sum(~np.isnan(atom_dataset['C4_NMR_shift'])), sum(~np.isnan(atom_dataset['H5_NMR_shift'])))
    print(sum(~np.isnan(atom_dataset['C1_Vbur'])), sum(~np.isnan(atom_dataset['C4_Vbur'])))
    print(sum(~np.isnan(atom_dataset['O3_NBO_charge'])), sum(~np.isnan(atom_dataset['C1_NBO_charge'])), sum(~np.isnan(atom_dataset['C4_NBO_charge'])), sum(~np.isnan(atom_dataset['O2_NBO_charge'])), sum(~np.isnan(atom_dataset['H5_NBO_charge'])))
    
    atom_dataset_reduced = atom_dataset[atom_dataset.atom_type.notnull()]
    groups = atom_dataset_reduced.groupby(['Name_int', 'Conf'])
    H5 = []
    C4 = []
    C1 = []
    O2 = []
    O3 = []
    for g in groups:
        if 'H5' not in list(g[1].atom_type):
            H5.append(g[0][0])
        if 'C4' not in list(g[1].atom_type):
            C4.append(g[0][0])
        if 'C1' not in list(g[1].atom_type):
            C1.append(g[0][0])
        if 'O2' not in list(g[1].atom_type):
            O2.append(g[0][0])
        if 'O3' not in list(g[1].atom_type):
            O3.append(g[0][0])
    print(H5, C4, C1, O2, O3) # these groups should be empty
    print(len(set(H5)), len(set(C4)), len(set(C1)), len(set(O2)), len(set(O3))) # these should all be zero
    
    bad_IDs = set(H5).union(set(C4)).union(set(C1)).union(set(O2)).union(set(O3))
    print(bad_IDs)
    
    # removing molecules with missing atom data (for convenience, and quality control)
    mol_dataset = mol_dataset[~mol_dataset['Name_int'].isin(bad_IDs)].reset_index(drop = True)
    atom_dataset = atom_dataset[~atom_dataset['Name_int'].isin(bad_IDs)].reset_index(drop = True)
    bond_dataset = bond_dataset[~bond_dataset['Name_int'].isin(bad_IDs)].reset_index(drop = True)
    
    
    # QCing bond dataset
    
    # Ideally this would be 0. But, there are 94 mol blocks that don't EXACTLY match; I think this is caused by stereochemistry recognition, not different coordinates
    print(len(set(mol_dataset.mol_block).union(set(bond_dataset.mol_block))) - len(set(mol_dataset.mol_block)))
    
    # these should all be the same
    print(sum(~np.isnan(bond_dataset['IR_freq'])), sum(~np.isnan(bond_dataset['Sterimol_L(Å)_morfeus'])), sum(~np.isnan(bond_dataset['Sterimol_B5(Å)_morfeus'])), sum(~np.isnan(bond_dataset['Sterimol_B1(Å)_morfeus'])))
    
    # making sure each conformer has consistent atom indexing 
    groups = bond_dataset[bond_dataset['IR_freq'].notnull()].groupby('Name_int')
    for g in groups:
        assert (len(np.unique(g[1]['bond_atom_tuple']))) == 1
    
    # remove molecules that don't have complete data 
    groups = bond_dataset.groupby('Name_int')
    bad_IDs = [] 
    for g in groups:
        if np.sum(~np.isnan(g[1]['IR_freq'])) != np.sum(~np.isnan(g[1]['Sterimol_B5(Å)_morfeus'])):
            bad_IDs.append(g[0])
    print('removed molecules due to missing bond data:', set(bad_IDs))
    
    # removing molecules with missing bond data (for convenience, and quality control)
    mol_dataset = mol_dataset[~mol_dataset['Name_int'].isin(bad_IDs)].reset_index(drop = True)
    atom_dataset = atom_dataset[~atom_dataset['Name_int'].isin(bad_IDs)].reset_index(drop = True)
    bond_dataset = bond_dataset[~bond_dataset['Name_int'].isin(bad_IDs)].reset_index(drop = True)
    
    
    
    # making sure atom indices are consistent across conformations (this checks for both atom/bond datasets)
    groups_QC = atom_dataset[~np.isnan(atom_dataset['NBO_charge'])].groupby('Name_int')
    for g in tqdm(groups_QC):
        assert len(np.unique(g[1][g[1]['atom_type'] == 'O3'].atom_index)) == 1
        assert len(np.unique(g[1][g[1]['atom_type'] == 'O2'].atom_index)) == 1
        assert len(np.unique(g[1][g[1]['atom_type'] == 'C4'].atom_index)) == 1
        assert len(np.unique(g[1][g[1]['atom_type'] == 'C1'].atom_index)) == 1
        assert len(np.unique(g[1][g[1]['atom_type'] == 'H5'].atom_index)) == 1
    
    
    # removing certain molecules manually
    remove_names = ['Ac67', 'Ac390', 'Ac4941', 'Ac6231', 'Ac5895', 'Ac67', 'Ac390', 'Ac801',  'Ac4045']
    mol_dataset = mol_dataset[~mol_dataset['Name'].isin(remove_names)].reset_index(drop = True)
    bond_dataset = bond_dataset[~bond_dataset['name'].isin(remove_names)].reset_index(drop = True)
    atom_dataset = atom_dataset[~atom_dataset['name'].isin(remove_names)].reset_index(drop = True)
    
    
    # step 7 computing modeling dataset
        
    # computing regression targets for each conformer; will be aggregated (boltz, max, min) later
    if property_type == 'mol':
        y_dataset = mol_dataset # = mol_dataset[mol_dataset[property_name].notnull()]
    elif property_type == 'atom':
        y_dataset = atom_dataset[atom_dataset[property_name].notnull()]
        assert list(y_dataset.Name_int) == list(mol_dataset.Name_int)
        assert list(y_dataset.Conf) == list(mol_dataset.Conf)
        y_dataset['min_E_conf'] = list(mol_dataset['min_E_conf'])
        y_dataset['max_E_conf'] = list(mol_dataset['max_E_conf'])
    elif property_type == 'bond':
        y_dataset = bond_dataset[bond_dataset[property_name].notnull()]
        assert list(y_dataset.Name_int) ==  list(mol_dataset.Name_int)
        assert list(y_dataset.Conf) ==  list(mol_dataset.Conf)
        y_dataset['min_E_conf'] = list(mol_dataset['min_E_conf'])
        y_dataset['max_E_conf'] = list(mol_dataset['max_E_conf'])
    
    # boltzmann weights
    if property_aggregation in ['boltz', 'min', 'max', 'min_E', 'max_E']:
        y_dataset[boltzmann_aggregation] = list(mol_dataset[boltzmann_aggregation])
    
    # determining which conformer has the minimum/maximum property-of-interest
    #if conformer_type in ['min_DFT', 'max_DFT']: # ----------------
    groups = y_dataset.groupby('Name_int', sort = False)
    select_by_min_property = np.zeros(len(y_dataset), dtype = int)
    select_by_max_property = np.zeros(len(y_dataset), dtype = int)
    pointer = 0
    for g in tqdm(groups):
        name = g[0]
        confs = np.array(g[1]['Conf'])
        group_property = np.array(g[1][property_name])
        select_min_conf = np.zeros(len(g[1]), dtype = int)
        select_max_conf = np.zeros(len(g[1]), dtype = int)
        select_min_conf[np.argmin(group_property)] = 1 # min
        select_max_conf[np.argmax(group_property)] = 1 #max
        select_by_min_property[pointer:pointer+len(g[1])] = select_min_conf
        select_by_max_property[pointer:pointer+len(g[1])] = select_max_conf
        pointer += len(g[1])
    # this only works if y_dataset and mol_dataset have 1-to-1 correspondence in their conformers (assert statements above)
    assert len(mol_dataset) == len(select_by_min_property) == len(select_by_max_property) 
    mol_dataset['min_conf'] = select_by_min_property
    mol_dataset['max_conf'] = select_by_max_property
    y_dataset['min_conf'] = select_by_min_property
    y_dataset['max_conf'] = select_by_max_property
    # ----------------
    
    if property_aggregation in ['boltz', 'min', 'max', 'min_E', 'max_E']: #else 'none'
        
        # aggregating properties (ignored if property_aggregation == 'none'; e.g. if computing per-conformer properties)
        if property_aggregation == 'boltz':
            y_data = pd.DataFrame(y_dataset.groupby('Name_int').apply(lambda x: np.sum(np.array(x[property_name]) * np.array(x[boltzmann_aggregation]))), columns = ['y'])
        if property_aggregation == 'min':
            y_data = pd.DataFrame(y_dataset.groupby('Name_int').apply(lambda x: np.min(x[property_name])), columns = ['y'])
        if property_aggregation == 'max':
            y_data = pd.DataFrame(y_dataset.groupby('Name_int').apply(lambda x: np.max(x[property_name])), columns = ['y'])
        if property_aggregation == 'min_E':
            y_data = pd.DataFrame(y_dataset.groupby('Name_int').apply(lambda x: x[x['min_E_conf'] == 1][property_name]).droplevel(1)).rename(columns = {property_name: 'y'})
            #y_data = pd.DataFrame(y_dataset.groupby('Name_int').apply(lambda x: x[x['min_E_conf'] == 1][property_name]).droplevel(1), columns = ['y'])
        if property_aggregation == 'max_E':
            y_data = pd.DataFrame(y_dataset.groupby('Name_int').apply(lambda x: x[x['max_E_conf'] == 1][property_name]).droplevel(1)).rename(columns = {property_name: 'y'})
            #y_data = pd.DataFrame(y_dataset.groupby('Name_int').apply(lambda x: x[x['max_E_conf'] == 1][property_name]).droplevel(1), columns = ['y'])
        
        # selecting/creating conformers used as modeling input
        if conformer_type == 'min_E_DFT': # assumes N_conformers = 1
            dataset = mol_dataset[mol_dataset['min_E_conf'] == 1][['Name', 'Conf', 'Name_int', 'mol_block', 'min_E_conf', 'max_E_conf']].reset_index(drop = True)
            dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(dataset['mol_block'])]
        elif conformer_type == 'max_E_DFT': # assumes N_conformers = 1
            dataset = mol_dataset[mol_dataset['max_E_conf'] == 1][['Name', 'Conf', 'Name_int', 'mol_block', 'max_E_conf', 'min_E_conf']].reset_index(drop = True)
            dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(dataset['mol_block'])]
        elif conformer_type == 'min_DFT': # assumes N_conformers = 1
            dataset = mol_dataset[mol_dataset['min_conf'] == 1][['Name', 'Conf', 'Name_int', 'mol_block', 'min_E_conf', 'max_E_conf','min_conf']].reset_index(drop = True)
            dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(dataset['mol_block'])]
        elif conformer_type == 'max_DFT': # assumes N_conformers = 1
            dataset = mol_dataset[mol_dataset['max_conf'] == 1][['Name', 'Conf', 'Name_int', 'mol_block', 'min_E_conf', 'max_E_conf', 'max_conf']].reset_index(drop = True)
            dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(dataset['mol_block'])]
            
        else: # otherwise, like if we want to use a rdkit or xtb or (random) dft conformer, we return the min_E_conf as a dummy conformer, which will be replaced later
            # these will all have the same Name_int and Conf. We replace these with randomly sampled conformers (rdkit, xtb, or DFT)
            min_E_dataset = mol_dataset[mol_dataset['min_E_conf'] == 1][['Name', 'Conf', 'Name_int', 'mol_block', 'min_E_conf']].reset_index(drop = True)
            min_E_dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(min_E_dataset['mol_block'])]
            dataset = pd.DataFrame(np.repeat(min_E_dataset.values, N_conformers, axis=0))
            dataset.columns = min_E_dataset.columns
        
        """
        
        elif conformer_type == 'random_DFT':
            dataset = mol_dataset.groupby('Name_int').apply(lambda x: x.sample(min(N_conformers, len(x))))[['Name', 'Conf', 'Name_int', 'mol_block', 'min_E_conf', 'max_E_conf']].reset_index(drop = True)
            dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(dataset['mol_block'])]
        
        
        elif conformer_type == 'random_rdkit': # I'm moving other conformer generation outside this function. This function should ONLY return DFT conformers.
            # use minimum energy conformer as a template for rdkit conformer generation
            min_E_dataset = mol_dataset[mol_dataset['min_E_conf'] == 1][['Name', 'Conf', 'Name_int', 'mol_block', 'min_E_conf']].reset_index(drop = True) 
            # make exactly N_conformers rdkit conformers for each molecule (alternatively, we could mirror what we do for random_DFT, to keep dataset sizes consistent)
            dataset = pd.DataFrame(np.repeat(min_E_dataset.values, N_conformers, axis=0))
            dataset.columns = min_E_dataset.columns
            mols_temp_3D = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(dataset['mol_block'])]
            rdkit_mols = generate_rdkit_conformers_from_3D_template(mols_temp_3D)
            dataset['mols'] = rdkit_mols
        """
        
        # merging regression targets and selected molecules
        modeling_dataset = dataset.merge(y_data, on =['Name_int'])[['Name_int', 'Conf', 'mols', 'y']]
        
        if property_type == 'bond':
            modeling_dataset = modeling_dataset.merge(y_dataset[['bond_atom_tuple', 'Name_int']].drop_duplicates(), on = 'Name_int').reset_index(drop = True)
        elif property_type == 'atom':
            modeling_dataset = modeling_dataset.merge(y_dataset[['atom_index', 'Name_int']].drop_duplicates(), on = 'Name_int').reset_index(drop = True)
    
            
    elif property_aggregation == 'none':
        #predicting per-conformer properties, using DFT conformers (only)
        # either use/predict all DFT (setting N_conformers to large number), a random subset of DFT, or the minimum energy conformation
            
        if conformer_type == 'min_E_DFT':
            modeling_dataset = y_dataset[y_dataset['min_E_conf'] == 1].reset_index(drop = True)
            modeling_dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(modeling_dataset['mol_block'])]
        elif conformer_type == 'max_E_DFT':
            modeling_dataset = y_dataset[y_dataset['max_E_conf'] == 1].reset_index(drop = True)
            modeling_dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(modeling_dataset['mol_block'])]
        elif conformer_type == 'min_DFT':
            modeling_dataset = y_dataset[y_dataset['min_conf'] == 1].reset_index(drop = True)
            modeling_dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(modeling_dataset['mol_block'])]
        elif conformer_type == 'max_DFT':
            modeling_dataset = y_dataset[y_dataset['max_conf'] == 1].reset_index(drop = True)
            modeling_dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(modeling_dataset['mol_block'])]
        elif conformer_type == 'random_DFT':
            modeling_dataset = y_dataset.groupby('Name_int').apply(lambda x: x.sample(min(N_conformers, len(x)))).reset_index(drop = True)
            modeling_dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(modeling_dataset['mol_block'])]
        
        modeling_dataset['y'] = np.array(modeling_dataset[property_name])
        
        # we also want to include the 'Conf' label, for swapping in 1-to-1 associated conformers re-"optimized" at MMFF or XTB level
        if property_type == 'mol':
            modeling_dataset = modeling_dataset[['Name_int', 'Conf', 'mols', 'y']]
        elif property_type == 'atom':
            modeling_dataset = modeling_dataset[['Name_int', 'Conf', 'mols', 'atom_index', 'y']]
        elif property_type == 'bond':
            modeling_dataset = modeling_dataset[['Name_int', 'Conf', 'mols', 'bond_atom_tuple', 'y']]
            
    
    # step 8 removing hydrogens
    
    # if we choose to remove Hs (for modeling), then we require that the molecules' atomic indices do not change upon H removal.
    # While we could explicitly change the atom indicing to make everthing match, it's convenient right now to just remove the few bad molecules.
    # actually, for consistency, we're always going to do this even if keep_explicit_hydrogens == True
    #if (keep_explicit_hydrogens == False):
    mols_no_Hs = [rdkit.Chem.RemoveHs(m) for m in list(modeling_dataset.mols)]
    mols_with_Hs = list(modeling_dataset.mols)
    remove_mols = []
    remove_mols_Name_int = []
    for i in tqdm(range(len(mols_no_Hs))):
        m_nH = mols_no_Hs[i]
        m_H = mols_with_Hs[i]
        m_canon = rdkit.Chem.MolFromSmiles(rdkit.Chem.MolToSmiles(m_H))
        m_canon = rdkit.Chem.AddHs(m_canon)
        for j in range(m_nH.GetNumAtoms()):
            if m_nH.GetAtomWithIdx(j).GetAtomicNum() != m_H.GetAtomWithIdx(j).GetAtomicNum():
                remove_mols.append(i)
                remove_mols_Name_int.append(modeling_dataset.iloc[i].Name_int)
                break
    modeling_dataset['mols_noHs'] = mols_no_Hs
    print('removed molecules due to atom indexing mismatch:', set(remove_mols_Name_int))  
    modeling_dataset = modeling_dataset[~modeling_dataset['Name_int'].isin(remove_mols_Name_int)].reset_index(drop = True)
    
    modeling_dataset['active_conformer'] = np.ones(len(modeling_dataset), dtype = int)
    
    
    # removing high energy ensembles or clustered ensembles
    if remove_clustered_ensembles or remove_high_energy_ensembles:
        modeling_dataset = modeling_dataset[~modeling_dataset['Name_int'].isin(remove_groups)].reset_index(drop = True)
    
    
    return modeling_dataset


    """
    # ------------------------------------
    if sample_extra_conformers is not None:
        
        active_labeling_map = {
                'max_DFT': 'max_conf',
                'min_DFT': 'min_conf',
                'max_E_DFT': 'max_E_conf',
                'min_E_DFT': 'min_E_conf',
            }
            
        if sample_extra_conformers[1] == 'DFT':
            
            assert conformer_type in active_labeling_map # only permitting this option if the "active labeled" conformer is the true DFT active, not a random rdkit or random DFT. Hence, if we want to use DFT decoys, then the "active" must be the active DFT conformer.
            
            # sampling other DFT conformers that AREN'T the active.
    
            decoy_dataset = y_dataset.groupby('Name_int').apply(lambda x: x[x[active_labeling_map[conformer_type]] == 0].sample(min(sample_extra_conformers[0], len(x) - 1)))
            decoy_dataset = decoy_dataset.reset_index(drop = True) #decoy_dataset[['Conf', 'Name_int', 'mol_block', 'min_E_conf', 'max_E_conf', 'min_conf', 'max_conf']].reset_index(drop = True)
            
            decoy_dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in list(decoy_dataset['mol_block'])]
            
            if (keep_explicit_hydrogens == False):
                mols_no_Hs = [rdkit.Chem.RemoveHs(m) for m in list(decoy_dataset.mols)]    
                decoy_dataset['mols_noHs'] = mols_no_Hs
                
            decoy_dataset = decoy_dataset.merge(modeling_dataset[['Name_int', 'y']], on = 'Name_int').reset_index(drop = True)
            decoy_dataset = decoy_dataset[list(modeling_dataset.columns)]
            
            decoy_dataset['active_conformer'] = np.zeros(len(decoy_dataset), dtype = int)
        
        
        elif sample_extra_conformers[1] == 'rdkit':
            # in this case, the 'active conformer' label will be meaningless, but it allows us to use this function for set aggregation of random rdkit conformers
            # we could also achieve the same effect by just using N_conformers > 1, ensemble_pooling = True, and sample_extra_conformers[0] = 0. But this is perhaps more confusing.
            if conformer_type == 'random_rdkit':
                assert N_conformers == 1
            
            # use random conformer as a template for rdkit conformer generation
            template_dataset = modeling_dataset.groupby('Name_int', sort = False).apply(lambda x: x.sample(1)).reset_index(drop = True)
            
            # make exactly N_conformers rdkit conformers for each molecule
            decoy_dataset = pd.DataFrame(np.repeat(template_dataset.values, sample_extra_conformers[0], axis=0))
            
            decoy_dataset.columns = template_dataset.columns
            mols_temp_3D = list(decoy_dataset['mols'])
            rdkit_mols = generate_rdkit_conformers_from_3D_template(mols_temp_3D)
            decoy_dataset['mols'] = rdkit_mols
            
            if (keep_explicit_hydrogens == False):
                mols_no_Hs = [rdkit.Chem.RemoveHs(m) for m in list(decoy_dataset.mols)]    
                decoy_dataset['mols_noHs'] = mols_no_Hs
        
            decoy_dataset['active_conformer'] = np.zeros(len(decoy_dataset), dtype = int)
    
    # ------------------------------------
        
    modeling_dataset['active_conformer'] = np.ones(len(modeling_dataset), dtype = int)
    if sample_extra_conformers is not None:
        modeling_dataset = pd.concat([modeling_dataset, decoy_dataset]).sort_values(by = ['Name_int', 'active_conformer']).reset_index(drop = True)
    """

    

def create_datasets_acids_massive_ensembles(property_aggregation, property_type, property_name, N_conformers):
    assert property_aggregation in ['min', 'max']
    assert property_type == 'bond'
    assert property_name in ['Sterimol_L(Å)_morfeus', 'Sterimol_B5(Å)_morfeus', 'Sterimol_B1(Å)_morfeus']
    
    y_label = property_name
    
    combined_df = pd.read_pickle('acid_data/massive_ensembles/20230512_combined_massive_ensembles_df.pkl')
    
    if y_label == 'Sterimol_L(Å)_morfeus':
        combined_df_filtered = combined_df[combined_df.filtered_Sterimol_L].reset_index(drop = True)
    if y_label == 'Sterimol_B5(Å)_morfeus':
        combined_df_filtered = combined_df[combined_df.filtered_Sterimol_B5].reset_index(drop = True)
    if y_label == 'Sterimol_B1(Å)_morfeus':
        combined_df_filtered = combined_df[combined_df.filtered_Sterimol_B1].reset_index(drop = True)
    
    select_columns = ['bond_index', 'Sterimol_L(Å)_morfeus', 'Sterimol_B5(Å)_morfeus', 'Sterimol_B1(Å)_morfeus', 'Name_int']
    
    groups = combined_df_filtered.groupby('Name_int', sort = False)
    
    min_Sterimol_L_indices = []
    max_Sterimol_L_indices = []
    
    min_Sterimol_B5_indices = []
    max_Sterimol_B5_indices = []
    
    min_Sterimol_B1_indices = []
    max_Sterimol_B1_indices = []
    
    for g in tqdm(groups):
    
        min_Sterimol_L = np.argmin(np.array(g[1]['Sterimol_L(Å)_morfeus']))
        min_Sterimol_L_indices.append((g[1].index[min_Sterimol_L]))
        
        max_Sterimol_L = np.argmax(np.array(g[1]['Sterimol_L(Å)_morfeus']))
        max_Sterimol_L_indices.append((g[1].index[max_Sterimol_L]))
        
        
        min_Sterimol_B5 = np.argmin(np.array(g[1]['Sterimol_B5(Å)_morfeus']))
        min_Sterimol_B5_indices.append((g[1].index[min_Sterimol_B5]))
    
        max_Sterimol_B5 = np.argmax(np.array(g[1]['Sterimol_B5(Å)_morfeus']))
        max_Sterimol_B5_indices.append((g[1].index[max_Sterimol_B5]))
        
            
        min_Sterimol_B1 = np.argmin(np.array(g[1]['Sterimol_B1(Å)_morfeus']))
        min_Sterimol_B1_indices.append((g[1].index[min_Sterimol_B1]))
        
        max_Sterimol_B1 = np.argmax(np.array(g[1]['Sterimol_B1(Å)_morfeus'])) 
        max_Sterimol_B1_indices.append((g[1].index[max_Sterimol_B1]))
    
    
    if (y_label == 'Sterimol_L(Å)_morfeus') & (property_aggregation == 'min'):
        min_Sterimol_L_mol_df = combined_df_filtered.iloc[min_Sterimol_L_indices][['mol_block', 'bond_index', 'Sterimol_L(Å)_morfeus', 'Name_int', 'Conf', 'rel_energy']]
        modeling_dataset = min_Sterimol_L_mol_df.rename(columns = {y_label: 'y'})
    if (y_label == 'Sterimol_L(Å)_morfeus') & (property_aggregation == 'max'):
        max_Sterimol_L_mol_df = combined_df_filtered.iloc[max_Sterimol_L_indices][['mol_block', 'bond_index', 'Sterimol_L(Å)_morfeus', 'Name_int', 'Conf','rel_energy']]
        modeling_dataset = max_Sterimol_L_mol_df.rename(columns = {y_label: 'y'})
        
    if (y_label == 'Sterimol_B5(Å)_morfeus') & (property_aggregation == 'min'):
        min_Sterimol_B5_mol_df = combined_df_filtered.iloc[min_Sterimol_B5_indices][['mol_block', 'bond_index', 'Sterimol_B5(Å)_morfeus', 'Name_int', 'Conf','rel_energy']]
        modeling_dataset = min_Sterimol_B5_mol_df.rename(columns = {y_label: 'y'})
    if (y_label == 'Sterimol_B5(Å)_morfeus') & (property_aggregation == 'max'):
        max_Sterimol_B5_mol_df = combined_df_filtered.iloc[max_Sterimol_B5_indices][['mol_block', 'bond_index', 'Sterimol_B5(Å)_morfeus', 'Name_int', 'Conf','rel_energy']]
        modeling_dataset = max_Sterimol_B5_mol_df.rename(columns = {y_label: 'y'})
    
    if (y_label == 'Sterimol_B1(Å)_morfeus') & (property_aggregation == 'min'):
        min_Sterimol_B1_mol_df = combined_df_filtered.iloc[min_Sterimol_B1_indices][['mol_block', 'bond_index', 'Sterimol_B1(Å)_morfeus', 'Name_int', 'Conf','rel_energy']]
        modeling_dataset = min_Sterimol_B1_mol_df.rename(columns = {y_label: 'y'})
    if (y_label == 'Sterimol_B1(Å)_morfeus') & (property_aggregation == 'max'):
        max_Sterimol_B1_mol_df = combined_df_filtered.iloc[max_Sterimol_B1_indices][['mol_block', 'bond_index', 'Sterimol_B1(Å)_morfeus', 'Name_int', 'Conf','rel_energy']]
        modeling_dataset = max_Sterimol_B1_mol_df.rename(columns = {y_label: 'y'})
        
        
    modeling_dataset['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in modeling_dataset['mol_block']]
    modeling_dataset['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in modeling_dataset['mols']]
    modeling_dataset = modeling_dataset.drop(columns = ['mol_block', 'rel_energy'])
    
    
    bond_indices = list(modeling_dataset.bond_index)
    bond_dataset_atom_tuple = []
    for i,m in tqdm(enumerate(list(modeling_dataset.mols)), total = len(modeling_dataset)):
        b = m.GetBondWithIdx(int(bond_indices[i]))
        bond_tuple = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        bond_dataset_atom_tuple.append(bond_tuple)
    modeling_dataset['bond_atom_tuple'] = bond_dataset_atom_tuple
    
    bond_dataset_atom_tuple_manual = []
    for m in modeling_dataset.mols:
        C1,_,_,C4,_ = get_COOH_idx(deepcopy(m))
        bond_dataset_atom_tuple_manual.append(tuple(sorted([C1, C4])))
    assert bond_dataset_atom_tuple_manual == bond_dataset_atom_tuple
    
    modeling_dataset = modeling_dataset.drop(columns = ['bond_index'])
    
    modeling_dataset['active_conf'] = np.ones(len(modeling_dataset), dtype = int)
    
    modeling_dataset = pd.DataFrame(np.repeat(modeling_dataset.values, N_conformers, axis=0), columns = modeling_dataset.columns)
        
    return modeling_dataset