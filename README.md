# Deep_SOC_estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1109%2FMETROXRAINE62247.2024.10795955-blue)](https://doi.org/10.1109/METROXRAINE62247.2024.10795955)

Python code to reproduce the results of the paper *Temperature-Dependent State of Charge Estimation for Electric Vehicles Based on a Machine Learning Approach* by Simone Barcellona, Loris Cannelli, Lorenzo Codecasa, Silvia Colnago, Loredana Cristaldi, Christian Laurano, and Gabriele Maroni.

Published in the *2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)*.

Repository URL: [https://github.com/gabribg88/Deep_SOC_estimation](https://github.com/gabribg88/Deep_SOC_estimation)

## Overview
This repository implements a machine learning workflow for estimating the absolute state of discharge `q` and the state of charge `SOC` of a lithium-ion battery from a single pair of measurements:

- open-circuit voltage `V_oc`
- battery temperature `T_avg`

The model is a feed-forward neural network trained on experimentally measured OCV-q curves collected at different temperatures. Model selection and evaluation follow a nested leave-one-curve-out cross-validation strategy, matching the practical case in which predictions must generalize to previously unseen temperature conditions.

## Abstract
Lithium-ion batteries are extensively used in electric vehicles and many other energy storage applications. Reliable estimation of the state of charge (SOC) is therefore essential for safe and efficient battery management. In this work, we study the temperature dependence of the open-circuit-voltage (OCV) versus absolute state-of-discharge relationship and develop a machine learning approach to estimate battery SOC from a single pair of measurements: OCV and battery temperature. The implemented model is a feed-forward neural network trained on experimentally measured OCV-q curves collected at different temperatures. Model selection and performance assessment are carried out with a nested leave-one-curve-out cross-validation strategy, which reflects the practical setting in which predictions must be made at previously unseen temperatures. The repository includes the notebooks and source code used to reproduce the reported curve reconstruction, SOC estimation, and temperature sensitivity analysis results.

## Repository Structure
| Path | Description |
|---|---|
| `paper/cannelli2024b.pdf` | Published paper describing the method and reported results. |
| `data/OCV_temperature.mat` | Experimental dataset used by the active workflow. |
| `notebooks/01_nested_logo_training_eval.ipynb` | Main notebook for dataset preparation, nested LOGO training, curve reconstruction, and q/SOC error analysis. |
| `notebooks/02_temperature_sensitivity_pdp.ipynb` | Sensitivity-analysis notebook for temperature sweeps and `q(V_min)` interpretation. |
| `src/train.py` | Training loop and fold execution helper. |
| `src/nn_utils.py` | Neural network definitions, seeding, early stopping, and figure export helpers. |
| `src/plot_config.py` | Plot styling and shared plotting configuration. |
| `figures/pdf/` | Exported figures aligned with the paper workflow. |
| `archive/` | Legacy material kept out of the active workflow. |
| `scratch/` | Disposable local artifacts not needed for reproduction. |

## Getting the Code
Clone the repository with git:

```bash
git clone https://github.com/gabribg88/Deep_SOC_estimation.git
cd Deep_SOC_estimation
```

Alternatively, download the repository as a zip archive from GitHub.

## Requirements
Use a dedicated Python environment. A typical `conda` setup is:

```bash
conda create --name <env_name> python=3.11
conda activate <env_name>
pip install -r requirements.txt
```

The required Python dependencies are listed in [requirements.txt](requirements.txt).

## Data Availability
The active workflow expects the dataset:

```text
data/OCV_temperature.mat
```

The repository ignores `.mat` files through `.gitignore`, so the dataset may need to be added locally before running the notebooks if it is not distributed with the published repository or release assets.

## Reproducing the Results
After activating the environment, start Jupyter:

```bash
jupyter lab
```

Then run the notebooks in numerical order.

### 1. Training and Evaluation
```text
notebooks/01_nested_logo_training_eval.ipynb
```
This notebook:
- builds the modeling dataset from the measured OCV-q curves
- performs nested leave-one-curve-out cross-validation
- reconstructs the OCV-q curves
- evaluates absolute errors on `q` and `SOC`

### 2. Temperature Sensitivity Analysis
```text
notebooks/02_temperature_sensitivity_pdp.ipynb
```
This notebook:
- retrains the neural network using the selected configuration
- performs the temperature sensitivity analysis
- generates the `q(V_min)` versus temperature curve

Generated figures are saved in `figures/pdf/`.

## Notes
This repository was cleaned up from a previously mixed workspace. The current top-level structure documents only the active SOC-estimation workflow associated with [paper/cannelli2024b.pdf](paper/cannelli2024b.pdf).

## Citation
If you use this code or the associated results in your research, please cite:

```bibtex
@inproceedings{Barcellona2024TemperatureDependentSOC,
  title={Temperature-Dependent State of Charge Estimation for Electric Vehicles Based on a Machine Learning Approach},
  author={Barcellona, Simone and Cannelli, Loris and Codecasa, Lorenzo and Colnago, Silvia and Cristaldi, Loredana and Laurano, Christian and Maroni, Gabriele},
  booktitle={2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)},
  year={2024},
  doi={10.1109/METROXRAINE62247.2024.10795955}
}
```

## License
This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE).
