#!/bin/bash

#SBATCH -o job_logs/eval_benchmark.sh.log-%j
#SBATCH -c 4

# Loading the required module
source /etc/profile
module load anaconda/2022a

source activate 3D_GNNs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python evaluate_acids_benchmark.py $output_dir $property_type $model_type $keep_explicit_hydrogens $ablate_3D $use_atom_features $ensemble_pooling $ensembles_contain_actives $pretrained_path
