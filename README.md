# Profile Omnifold
This repository contains the codebase for the **Profile OmniFold** algorithm.

**Note:** This codebase is actively under development.


## Overview
The repository currently includes the following files:

* **`profile_omnifold.py`** 
  Implements the main methods for running both the Profile OmniFold algorithm and the original OmniFold algorithm. The key functions are:
  1. `omnifold`: the original OmniFold algorithm.
  2. `profile_omnifold`: the Profile OmniFold algorithm.

* **`profile_omnifold_no_nn.py`**
  Contains the same methods as `profile_omnifold.py`, but replaces neural networks with non–neural network classifiers.

* **`utils.py`**
  A coollection of utility functions used across the codebase.

* **`Gaussian_example.ipynb`**
  A Jupyter notebook demonstrating the Profile Omnifold algorithm on a Gaussian example.

* **`OpenData_example.ipynb`**
  A Jupyter notebook demonstrating the Profile Omnifold algorithm on [CMS open dataset](https://energyflow.network/docs/datasets/).