# Adaptive, Hybrid Feature Selection

Python implementation of the Adaptive, Hybrid Feature Selection algorithm (AHFS), originally developed by [Viharos et al.](https://doi.org/10.1016/j.patcog.2021.107932)
For scientific or related inquiries, please contact [Dr. Zsolt János Viharos](https://sztaki.hun-ren.hu/en/organisation/departments/emi/zsolt-janos-viharos) and [Anh Tuan Hoang](https://sztaki.hun-ren.hu/en/organisation/departments/emi/anh-tuan-hoang).

## Getting started

### Requirements
- Windows or Linux-based platform
- Python version 3.11 or better
- *Optional:* CUDA 11.8 or better

### Installation

Install from PyPI via ```pip install ahfs```. It is recommended that you create a separate environment.

### Usage

You may run one of the preset configurations or run an instance with your own dataset and settings.

#### Presets

To run a preset configuration, first download the ```datasets``` folder from this repository into your working directory.
Secondly, import the desired configuration from ```utils.presets``` or use the example code found in ```utils.example```.
Run the configuration by invoking the ```run()``` method on the class instance.

Consult the [API documentation](https://ahfs.readthedocs.io/en/latest/) for further details.

#### Running your own instance

Consult the [API documentation](https://ahfs.readthedocs.io/en/latest/) for further details.

## FAQ

1. I get the warning message ***CUDA is not available! Using CPU..***
   - Re-install the torch package by following [these instructions](https://pytorch.org/get-started/locally/).
