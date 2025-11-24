# SpheroidPy
[![PyPI version](https://badge.fury.io/py/SpheroidPy.svg)](https://badge.fury.io/py/SpheroidPy)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

<img src="SpheroidPy/static/images/logo_full.png" width="200em" align="right" style="margin-left: 20px; margin-bottom: 10px;" />


`SpheroidPy` is a Python package designed for the management and analysis of _in-vitro_ spheroid data. It offers a framework to facilitate the handling, segmentation, and analysis of extensive collections of spheroid microscopy images. The package integrates mechanistic biophysical models enabling users to extract insights into the kinetics of underlying biological processes. While primarily developed and optimized for cancer spheroid proliferation and cytotoxicity assays, it can be adapted to other cell entities and three-dimensional assay systems.


## Applications

### Proliferation Assays
<img src="docs/source/_images/spheroid_growth_example_movie.gif" width="180px" align="right" style="margin-left: 50px; margin-bottom: 50px;" />

Proliferation assays are essential for studying growth kinetics in three-dimensional spheroid cultures. Besides simple growth curves, more complex behaviors - such as the emergence of a necrotic core at a critical radius $R_c$ or saturation at large sizes - can be assessed from such data.
`SpheroidPy` facilitates data import, spheroid segmentation, and comprehensive analysis of growth dynamics, including automated statistical evaluations.

### Cytotoxity Assays

To be implemented...

## Prerequisits

To ensure a clean and reproducible setup, we recommend installing the package inside a dedicated Conda environment. Begin by installing [Anaconda](https://www.anaconda.com/products/distribution), then create and activate a new environment using the commands below. This can be done in a Anaconda Prompt (Windows) or Terminal (Mac/Linux) and guarantees that all dependencies—including optional deep-learning frameworks—are isolated from your system installation.

```bash 
conda create -n spheroid python=3.10
conda activate spheroid
```

#### Optional: Installing Detectron2 for Deep-Learning-Based Segmentation

SpheroidPy supports segmentation pipelines that can optionally leverage Detectron2. Since Detectron2 is not available via standard PyPI and provides separate builds for CPU and GPU systems, it must be installed manually. Most users without a dedicated NVIDIA GPU will benefit from the CPU version, which is simpler to install and compatible across operating systems.

##### CPU version (recommended for most users):

```bash
# Install PyTorch 2.3
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu

# Install Detectron2 0.6
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.3/index.html
```

##### GPU version (if CUDA is available):

Choose the appropriate wheel for your PyTorch and CUDA version from the  [Detectron2 official installation guide](https://github.com/facebookresearch/detectron2). Follow the provided installation instructions for your platform and CUDA configuration.



## Installation

#### From Source
The package can be installed directly from the source. Therefore, this GitHub repository has to be downloaded.
After navigating to the folder containing package this can be done using pip:

```bash 
cd path/to/SpheroidPy   # navigate to the parent folder
pip install .           # install the package
```

This installs `SpheroidPy` along with all required dependencies, making it immediately available in your environment.

## Usage
After installation, `SpheroidPy` can be imported and utilized in Python scripts or Jupyter Notebooks. For the establishment of new workflows the usage of jupyter notebook is recommended. Examples and tutorials can be found in the respective [folder](https://github.com/CedHe/SpheroidPy/tree/master/examples). The following section will guide you through an examplary workflow.

```python
import SpheroidPy
```