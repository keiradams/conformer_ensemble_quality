# The Impact of Conformer Quality on Learned Representations of Molecular Conformer Ensembles
Code and data for reproducing the [preprint](https://arxiv.org/abs/2502.13220): Adams, K. and C.W. Coley, *The impact of conformer quality on learned representations of molecular conformer ensembles*, 2025.

## Environment
`environment.yml` contains the full conda environment. The primary dependencies are `torch-geometric==2.1.0` (& supporting packages like `torch-scatter==2.0.9`), `pytorch=1.13.0`, and `rdkit=2022.03.2`.

## Downloading data and trained model weights
Before reproducing any results, you will need to download the datasets (target properties and conformers) from this Dropbox link: .

This Dropbox also includes all the trained models and inference results that were reported in the preprint. Due to the numerous models trained and evaluated, the total size of this data is ~50GB.

## Retraining

`train_acids_benchmark.py` contains the training script used to train all models. It takes as input a list of command-line parameters (parsed by `argparse`) that define each model version as well as the composition of the corresponding training data. Each model is trained separately with separate calls to `train_acids_benchmark.py`. 

`job_submissions.py` contains the list of all parameters used for each model that is reported in the preprint. Calling `job_submissions.py` will submit multiple, independent jobs (through a SLURM system) for training each model.

## Evaluations
`evaluate_acids_benchmark.py` contains the evaluation script used to evaluate each model. 

Calling `submissions_evaluations_benchmark.py` will submit multiple, independent jobs (through a SLURM system) in order to evaluate each model that is reported in the preprint.


## Analyzing results
The notebooks `analysis_benchmark_20241109_figure{*}.ipynb` analyze all evaluation results (downloadable from the above Dropbox link) to reproduce the main-text figures that are reported in the preprint.


## License

This project is licensed under the MIT License -- see [LICENSE](./LICENSE) file for details.

## Citation
```bibtex
@misc{adams2025impactconformerqualitylearned,
      title={The impact of conformer quality on learned representations of molecular conformer ensembles}, 
      author={Keir Adams and Connor W. Coley},
      year={2025},
      eprint={2502.13220},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.13220}, 
}
```
