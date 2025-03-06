#!/bin/bash

#SBATCH -o job_logs/job.sh.log-%j
#SBATCH -c 4

# Loading the required module
source /etc/profile
module load anaconda/2022a

source activate 3D_GNNs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the script
python train_acids_benchmark.py $output_PATH $output_file $property_type $property_name $property_aggregation $conformer_type $N_conformers $keep_explicit_hydrogens $model_type $ablate_3D $pretrained_path $batch_size $lr $seed $ensemble_pooling $sample_extra_conformers_N $sample_extra_conformers_type $active_classification_loss $perturb_DFT_conformer $epoch $use_atom_features $train_fraction
