# Profile Omnifold
This repository contains codes for profile Omnifold.


## Overview
The repository includes the following files:

* `profile_omnifold.py`: Contains the main methods for running the Profile Omnifold algorithm, as well as the original Omnifold and its various versions. Speficifically, the following algorithms are included.
  1. `omnifold`: original Omnifold
  2. `penalized_profile_omnifold`: Profile Omnifold algorithm
  3. `nonparametric_profile_omnifold`: Profile Omnifold without parametrizing the response kernel
  4. `ad_hoc_penalized_profile_omnifold`: Profile Omnifold with additional penalty term added based on `nonparametric_profile_omnifold`
* `profile_omnifold_no_nn.py`: Provides the same methods as `profile_omnifold.py`, but without using neural networks for classifier training.
* `utils.py`: Utility functions.
* `profile_omnifold_demo.ipynb`: Jupyter notebook demonstrating the Profile Omnifold algorithm and comparing it with other methods.
* `simulation.ipynb`: Jupyter notebook that performs a simulation study for the Profile Omnifold algorithm.
