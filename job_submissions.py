import os

# Legend:
# outdir, prop_type, prop_name, aggregation, conformer_type, N_confs, Hs, model, ablate3D, pretrained, batch_size, lr, seed, ensemble_pooling, N_decoys, decoy_type, active_class_loss, DFT_perturbation, epoch, use_atom_features, train_fraction

# repeat for seeds 0, 1, 2
SEED = 0


params = [
    
    # training with 1 random rdkit conformer, with decreasing dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    # training with 1 random xtb conformer, with decreasing dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    # training with 1 random DFT conformer, with decreasing dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    
    # training with random rdkit conformers, with data augmentation
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    
    # training with random xtb conformers, with data augmentation
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0], # run with 4 cpus
    
    
    # training with random DFT conformers, with data augmentation
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    
    
    # training with ground-truth active conformers, with decreasing dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    
    # training with xtb perturbed ground-truth active conformers, with varying dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    
    # training with rdkit perturbed ground-truth active conformers, with varying dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    

    
    # training with ground-truth conformer and DFT decoys
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'DFT', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'DFT', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'DFT', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'DFT', 0] + ['none'] + [0] + [0] + [1.0],
    
    
    # training with xtb-perturbed GT conformer and XTB decoys
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'xtb', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'xtb', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'xtb', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'xtb', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    
    
    # training with MMFF94-perturbed GT conformer and MMFF94 decoys
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 9, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],

    
    
    # training with ensembles of rdkit or xtb conformers, with varying dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],

    
    
    
    # training with 1 random rdkit conformer, with decreasing dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    # training with 1 random xtb conformer, with decreasing dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    # training with 1 random DFT conformer, with decreasing dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'max', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_L', 'min', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    # training with random rdkit conformers, with data augmentation
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],

    
    
    # training with random xtb conformers, with data augmentation
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],


    # training with random DFT conformers, with data augmentation
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'max', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_L', 'min', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'max', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 5, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 10, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_augmented', 'bond', 'Sterimol_B5', 'min', 'random_DFT', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],

    
    
    # training with ground-truth active conformers, with decreasing dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.25],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_active', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [0.125],



    # training with (rdkit or xtb) perturbed ground-truth active conformers, with varying dataset sizes
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.5],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.5],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.5],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.25],

    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.25],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.25],
    
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_xtb'] + [0] + [0] + [0.125],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.125],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [0, 0, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [0.125],

    
    

    # training with GT conformer and DFT decoys, 19 decoys
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'DFT', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'DFT', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'DFT', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'DFT', 0] + ['none'] + [0] + [0] + [1.0],
    
    # training with xtb-perturbed GT conformer and XTB decoys, 19 decoys
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'xtb', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'xtb', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'xtb', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'xtb', 0] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    
    # training with MMFF94-perturbed GT conformer and MMFF94 decoys, 19 decoys
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'rdkit', 0] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    

    # training with GT conformer and DFT decoys, 19 decoys, active classification
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'DFT', 1] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'DFT', 1] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'DFT', 1] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'DFT', 1] + ['none'] + [0] + [0] + [1.0],
    
    # training with xtb-perturbed GT conformer and XTB decoys, 19 decoys, active classification
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'xtb', 1] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'xtb', 1] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'xtb', 1] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'xtb', 1] + ['dft_to_xtb'] + [0] + [0] + [1.0],
    
    # training with MMFF94-perturbed GT conformer and MMFF94 decoys, 19 decoys, active classification
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'rdkit', 1] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_L', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'rdkit', 1] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'max_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'rdkit', 1] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_decoys', 'bond', 'Sterimol_B5', 'none', 'min_DFT', 1, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 19, 'rdkit', 1] + ['dft_to_rdkit'] + [0] + [0] + [1.0],
    
    
    
    # training with ensembles of rdkit or xtb conformers, 20 conformers
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_rdkit', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_L', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'max', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],
    [f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble', 'bond', 'Sterimol_B5', 'min', 'random_xtb', 20, 2, 'DimeNetPlusPlus', 0, 'none', 64, 0.0001, SEED] + [1, 0, 'rdkit', 0] + ['none'] + [0] + [0] + [1.0],

]

# choose any subset of the above models/parameters to train, as otherwise this script will submit 100s of jobs at once
#params = params[0:10]

total_submitted = 0

output_files = []

for p_list in params:
    
    outdir, prop_type, prop_name, aggregation, conformer_type, N_confs, Hs, model, ablate3D, pretrained, batch_size, lr, seed, ensemble_pooling, N_decoys, decoy_type, active_class_loss, DFT_perturbation, epoch, use_atom_features, train_fraction = p_list
    
    f = f'{prop_type}_{prop_name}_{aggregation}_{conformer_type}_{N_confs}_{Hs}_{model}_{ablate3D}_{DFT_perturbation}_{int(train_fraction*1000)}_seed{seed}'
    
    if ensemble_pooling == 1:
        f += f'_ensemble_{N_decoys}_{decoy_type}_{active_class_loss}'
    elif ensemble_pooling > 1:
        f += f'_ensemble{ensemble_pooling}_{N_decoys}_{decoy_type}_{active_class_loss}'
    
    if (use_atom_features == 1):
        f += f'_features'
        
    p = p_list
    
    command = f'sbatch --export=output_PATH={p[0]},output_file={f},property_type={p[1]},property_name={p[2]},property_aggregation={p[3]},conformer_type={p[4]},N_conformers={p[5]},keep_explicit_hydrogens={p[6]},model_type={p[7]},ablate_3D={p[8]},pretrained_path={p[9]},batch_size={p[10]},lr={p[11]},seed={p[12]},ensemble_pooling={p[13]},sample_extra_conformers_N={p[14]},sample_extra_conformers_type={p[15]},active_classification_loss={p[16]},perturb_DFT_conformer={p[17]},epoch={p[18]},use_atom_features={p[19]},train_fraction={p[20]}'
    
    
    full_command = command + ' SBATCH_train_benchmark.sh'

    os.system(full_command)
    total_submitted +=1

print(f'{total_submitted} jobs submitted')

    

