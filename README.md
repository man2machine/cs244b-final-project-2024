# cs244b-final-project-2024

## Installation

The `conda_env.yml` file contains the necessary python packages to install using Anaconda.

## System Requirements

The experiments were performed on an HGX machine with 8 A100 GPUs with 80 GB each. To execute the FacTool application a minimum of 3 GPUs are needed, and this can be configured in `alto_py/alto_lib/apps/factool/setup.py`

## Usage

Run the `alto_py/scripts/factool/factool_client.ipynb` and the `alto_py/scripts/factool/factool_main_ray.ipynb` notebooks to run an experiment. Logs will be generated in the `logs/` folder at the root of the repository.