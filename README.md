# Profile Omnifold
This repository contains codes for Profile OmniFold algorithm.


## Overview
The repository includes the following files:

* `profile_omnifold.py`: Contains the main methods for running the Profile OmniFold algorithm, as well as the original Omnifold and its various versions. Speficifically, the following algorithms are included.
  1. `omnifold`: original OmniFold
  2. `profile_omnifold`: Profile OmniFold algorithm (relying on gradient of W function)
  3. `profile_omnifold_no_grad`: Profile OmniFold algorithm without using the gradient of W function
  4. `nonparametric_profile_omnifold`: Profile OmniFold without parametrizing the response kernel
  5. `ad_hoc_penalized_profile_omnifold`: Profile OmniFold with additional penalty term added based on `nonparametric_profile_omnifold`
* `profile_omnifold_no_nn.py`: Provides the same methods as `profile_omnifold.py`, but without using neural networks for classifier training.
* `utils.py`: Utility functions.
* `Gaussian_example.ipynb`: Jupyter notebook demonstrating the Profile Omnifold algorithm on a Gaussian Example.
* `OpenData_example.ipynb`: Jupyter notebook demonstrating the Profile Omnifold algorithm on [CMS open dataset](https://energyflow.network/docs/datasets/).
