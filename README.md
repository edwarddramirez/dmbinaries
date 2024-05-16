[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/license/mit) ![Python](https://img.shields.io/badge/python-3.9.16-blue.svg) ![Repo Size](https://img.shields.io/github/repo-size/edwarddramirez/allsky-point-source-detection) 

# dmbinaries
This repository contains the code used to produce the results shown in [arXiv:2209.08100](https://arxiv.org/abs/2209.08100). It is meant to act as a supplement focusing more on the actual structure of the code. Through our tutorial notebooks, we hope to help others in either better understanding our results or perform their own set of simulations for various purposes such as: setting their own constraints on subhalos, simulate binary evolution for some other purpose, or improve this code to set more robust limits. 

Please read the *README.md* file within each file for content information. We strongly encourage readers to read the paper before working with the code.

# Installation
Run the `environment.yml` file by running the following command on the main repo directory:
```
conda env create
```
The installation works for `conda==22.9.0`. 

## Directory Structure
### 1. Data
- Directory: data
  - Purpose: Show how we manipulated data to obtain the sample of binaries used to set constraints on dark matter subhalos
### 2. Binary Evolution Simulations
- Directory: evolution
  - Purpose: Simulate the evolution of binaries subject to tidal encounters with extended subhalos
### 3. Limits on Subhalos
- Directory: statistics
  - Purpose: Set limits on subhalos given the sample of binaries and the results of the binary evolution simulations
### 4. Supplementary Material
- Directory: supplement
    - Purpose: Provide more detailed discussions and derivations of various results
### 5. Figures
- Directory: figures
    - Purpose: Produces all the figures in the paper for readers that might benefit from manipulating the data
