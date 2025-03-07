import os

submissions = []
SEED = 0

parent_dir = f'jobs_acid_benchmark_seed{SEED}/jobs/'
dirs = next(os.walk(parent_dir))[1]
for d in dirs:
    assert ('DimeNetPlusPlus' in d) or ('GIN' in d)
    
    model_type = 'DimeNetPlusPlus' if ('DimeNetPlusPlus' in d) else 'GIN'
    use_atom_features = int(model_type == 'GIN')
    keep_explicit_hydrogens = 2
    ablate_3D = 0
    ensemble_pooling = 0
    ensembles_contain_actives = 0
    
    sub = [parent_dir + d, 'bond', model_type, keep_explicit_hydrogens, ablate_3D, use_atom_features, ensemble_pooling, ensembles_contain_actives, 'saved_models/model_best.pt']
    submissions.append(sub)
    
    
parent_dir = f'jobs_acid_benchmark_seed{SEED}/jobs_augmented/'
dirs = next(os.walk(parent_dir))[1]
for d in dirs:
    assert ('DimeNetPlusPlus' in d) or ('GIN' in d)
    
    model_type = 'DimeNetPlusPlus' if ('DimeNetPlusPlus' in d) else 'GIN'
    use_atom_features = int(model_type == 'GIN')
    keep_explicit_hydrogens = 2
    ablate_3D = 0
    ensemble_pooling = 0
    ensembles_contain_actives = 0
    
    sub = [parent_dir + d, 'bond', model_type, keep_explicit_hydrogens, ablate_3D, use_atom_features, ensemble_pooling, ensembles_contain_actives, 'saved_models/model_best.pt']
    submissions.append(sub)


parent_dir = f'jobs_acid_benchmark_seed{SEED}/jobs_perturbed/'
dirs = next(os.walk(parent_dir))[1]
for d in dirs:
    assert ('DimeNetPlusPlus' in d) or ('GIN' in d)
    
    model_type = 'DimeNetPlusPlus' if ('DimeNetPlusPlus' in d) else 'GIN'
    use_atom_features = int(model_type == 'GIN')
    keep_explicit_hydrogens = 2
    ablate_3D = 0
    ensemble_pooling = 0
    ensembles_contain_actives = 0
    
    sub = [parent_dir + d, 'bond', model_type, keep_explicit_hydrogens, ablate_3D, use_atom_features, ensemble_pooling, ensembles_contain_actives, 'saved_models/model_best.pt']
    submissions.append(sub)


parent_dir = f'jobs_acid_benchmark_seed{SEED}/jobs_active/'
dirs = next(os.walk(parent_dir))[1]
for d in dirs:
    assert ('DimeNetPlusPlus' in d) or ('GIN' in d)
    
    model_type = 'DimeNetPlusPlus' if ('DimeNetPlusPlus' in d) else 'GIN'
    use_atom_features = int(model_type == 'GIN')
    keep_explicit_hydrogens = 2
    ablate_3D = 0
    ensemble_pooling = 0
    ensembles_contain_actives = 0
    
    sub = [parent_dir + d, 'bond', model_type, keep_explicit_hydrogens, ablate_3D, use_atom_features, ensemble_pooling, ensembles_contain_actives, 'saved_models/model_best.pt']
    submissions.append(sub)

    
parent_dir = f'jobs_acid_benchmark_seed{SEED}/jobs_ensemble/'
dirs = next(os.walk(parent_dir))[1]
for d in dirs:
    assert ('DimeNetPlusPlus' in d) or ('GIN' in d)
    
    model_type = 'DimeNetPlusPlus' if ('DimeNetPlusPlus' in d) else 'GIN'
    use_atom_features = int(model_type == 'GIN')
    keep_explicit_hydrogens = 2
    ablate_3D = 0
    ensemble_pooling = 1
    ensembles_contain_actives = 0
    
    sub = [parent_dir + d, 'bond', model_type, keep_explicit_hydrogens, ablate_3D, use_atom_features, ensemble_pooling, ensembles_contain_actives, 'saved_models/model_best.pt']
    submissions.append(sub)

parent_dir = f'jobs_acid_benchmark_seed{SEED}/jobs_decoys/'
dirs = next(os.walk(parent_dir))[1]
for d in dirs:
    assert ('DimeNetPlusPlus' in d) or ('GIN' in d)
    
    model_type = 'DimeNetPlusPlus' if ('DimeNetPlusPlus' in d) else 'GIN'
    use_atom_features = int(model_type == 'GIN')
    keep_explicit_hydrogens = 2
    ablate_3D = 0
    ensemble_pooling = 1
    ensembles_contain_actives = 1
    
    sub = [parent_dir + d, 'bond', model_type, keep_explicit_hydrogens, ablate_3D, use_atom_features, ensemble_pooling, ensembles_contain_actives, 'saved_models/model_best.pt']
    submissions.append(sub)


print(len(submissions))

# choose any subset of the above models to evaluate
#submissions = submissions[0:10]

commands = []
for i in range(len(submissions)):
    p = submissions[i]
    
    c = f'sbatch --export=output_dir={p[0]},property_type={p[1]},model_type={p[2]},keep_explicit_hydrogens={p[3]},ablate_3D={p[4]},use_atom_features={p[5]},ensemble_pooling={p[6]},ensembles_contain_actives={p[7]},pretrained_path={p[8]}'
    
    commands.append(c)

for command in commands:
    full_command = command + ' SBATCH_eval_benchmark.sh'
    os.system(full_command)
