# Profile OmniFold

[![arXiv](https://img.shields.io/badge/arXiv-2512.07074-b31b1b.svg)](https://arxiv.org/abs/2512.07074)

This repository contains the code for **Profile OmniFold**, introduced in
[*Machine Learning-based Unfolding for Cross Section Measurements in the Presence of Nuisance Parameters*](https://arxiv.org/abs/2512.07074).

## Installation
Requires Python 3.11. Install the dependencies with:

```bash
pip install -r requirements.txt
```

**Note:** `profile_omnifold.py` uses the Keras 2 backend API, so TensorFlow must be older than 2.16 (Keras 3 removed this API). These TensorFlow versions support Python up to 3.11.

## Repository contents
* **`profile_omnifold.py`**: core implementation
  * `omnifold`: the original OmniFold algorithm
  * `profile_omnifold`: the Profile OmniFold algorithm
* **`utils.py`**: utility functions used across the codebase
* **`Gaussian_example.ipynb`**: Profile OmniFold on a Gaussian example
* **`OpenData_example.ipynb`**: Profile OmniFold on the [CMS Open Data](https://energyflow.network/docs/datasets/) dataset. On first run, the notebook downloads the simulated jet data through the `energyflow` package into `Dataset/CMS-Open-Data`; change `cache_dir` in the notebook to store it elsewhere.
