
Welcome to the AnaNC Model Repository!

This repository contains code and resources to reproduce the experiments presented in our paper "Continual Out-of-Distribution Detection with Analytic Neural Collapse". Follow the instructions below to set up the environment.

## Dependencies
1. Install Required Packages:
   - Install the dependencies using the provided `requirements.txt` file:
     pip install -r requirements.txt

## Quick Start
- The `ananc.py` file in the `models` folder is a standalone module for incremental learning and OOD detection. You can integrate this module directly into your own codebase by passing feature embeddings (from a pre-trained model) and class labels.  
  If you use this with any pre-trained model or dataset other than those in our experiments, we recommend tuning the regularization parameter (`reg`) in the range `[1e-2, 1e6]` with an increment factor of 10.

- You can optionally apply FSA (First Session Adaptation) before feature extraction to improve accuracy.

## Datasets
- The datasets must be downloaded from their official sources and placed in the `data/` folder.

- To evaluate the model and other baseline methods, use the provided scripts in the repository. These scripts reproduce our experimental results and can be adapted for custom datasets.


## Citation
If you find this repository useful, please cite our paper:

```
@inproceedings{momeni2025continual,
  title={Continual Out-of-Distribution Detection with Analytic Neural Collapse},
  author={Saleh Momeni, Changnan Xiao, and Bing Liu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
