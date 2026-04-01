# BayesLTV: Bayesian Modeling of Linear Time-Varying Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code and experimental notebooks for the paper, ["Bayesian Modeling and Estimation of Linear Time-Varying Systems using Neural Networks and Gaussian Processes](https://arxiv.org/abs/2507.12878)."

***

## Overview

The identification of **Linear Time-Varying (LTV)** systems from input-output data is a fundamental yet challenging ill-posed inverse problem. This work introduces a unified Bayesian framework that models the system's impulse response, $h(t, \tau)$, as a stochastic process. We decompose the response into a posterior mean and a random fluctuation term, a formulation that provides a principled approach for quantifying uncertainty and naturally defines a new system class we term **Linear Time-Invariant in Expectation (LTIE)**.

To perform inference, we leverage modern machine learning techniques, including Bayesian neural networks and Gaussian Processes, using scalable variational inference. This repository provides the tools to reproduce the key findings of our work.

### Key Features
* A unified Bayesian framework ($h = \mu + \mathcal{E}$) for modeling Linear Time-Varying systems.
* Implementation of three distinct experiments demonstrating the framework's versatility.
* Comparison against classical signal processing techniques in a simulated Ambient Noise Tomography (ANT) problem.
* Use of amortized variational inference with a Gaussian Process prior to track a continuously varying LTV impulse response.

***

## Repository Structure 📂
bayes-ltv/  
├── data/             # Data files for experiments  
├── paper/            # LaTeX source for the manuscript  
├── src/              # Source code, model implementations and notebooks  
├──── experiments/    # Notebooks and related code for experiments  
└──── models/         # LTIE and LTV model implementations    
***

## Experiments Overview 🔬

This repository includes the code for three core experiments detailed in the paper:

1.  **LTI System Identification**: Demonstrates how the Bayesian framework can robustly infer the properties of a
classic LTI system (and its uncertainty) from a single noisy observation pair. See the 
[lti_model_experiments.ipynb](https://github.com/yaniv-shulman/bayes-ltv/blob/main/src/experiments/ltie_estimation/ltie_model_experiments.ipynb) 
notebook for details.
2.  **Ambient Noise Tomography (ANT)**: A simulated geophysical application that compares the data efficiency of our
Bayesian model against the classical Cross-Correlation Function (CCF) stacking method. See the
[ant_synthetic_sine_pulses_curved_velocity.ipynb](https://github.com/yaniv-shulman/bayes-ltv/blob/main/src/experiments/ant/ant_synthetic_sine_pulses_curved_velocity.ipynb)
notebook for details.
3.  **LTV Impulse Response Regression**: Shows how a GP-regularized model can successfully track a continuously changing
impulse response, a highly ill-posed problem. See the
[ltv_model_experiments.ipynb](https://github.com/yaniv-shulman/bayes-ltv/blob/main/src/experiments/ltv_estimation/ltv_model_experiments.ipynb)
notebook for details.

***

## Getting Started 🚀

This project uses **Poetry** to manage dependencies and ensure a reproducible environment.

### Prerequisites

* **Python 3.11**
* **Poetry** (see [official documentation](https://python-poetry.org/docs/#installation) for installation instructions)
* An **NVIDIA GPU** with appropriate CUDA drivers is required to leverage `tensorflow-gpu` for the experiments.

### Installation

The simplest way to set up the project on a Linux-based system is by using the provided configuration script.

1.  Clone the repository
2.  **Run the configuration script:**
    This script will configure the necessary environment variables, instruct Poetry to use Python 3.11, install all dependencies from the `pyproject.toml` file, and activate the virtual environment.
    ```bash
    source configure.sh
    ```
    You are now ready to run the notebooks and experiments.

#### Manual Installation

If not using the script, you can set up the environment manually with these Poetry commands:

1.  **Select Python version:**
    ```bash
    poetry env use python3.11
    ```

2.  **Install dependencies:**
    ```bash
    poetry install --no-root
    ```

3.  **Activate the environment:**
    This command activates the Poetry-managed virtual environment, allowing you to run Python scripts and Jupyter
notebooks with the installed dependencies. Note that newer versions of Poetry may require you to install the poetry
shell plugin first.
       ```bash
       $(eval poetry env activate)
       ```
***

## Usage  notebooks

The easiest way to explore the results is through the **Jupyter notebooks** located in the `/src/experiments` directory.
These notebooks contain the analysis, data visualization, and figure generation for each experiment.
To regenerate the results from scratch, just rerun the notebooks.

***

## Citation

If you use this work, please cite the paper:
<pre><code class="language-bibtex">@misc{shulman2026bayesianmodelingestimationlinear,
      title={Bayesian Modeling and Estimation of Linear Time-Varying Systems using Neural Networks and Gaussian Processes}, 
      author={Yaniv Shulman},
      year={2026},
      eprint={2507.12878},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2507.12878}, 
}</code></pre>
