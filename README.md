<div align="center">
    <img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/logo_with_text.png" width=600>

**Bulldozer, a DTM extraction tool requiring only a DSM as input.**

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Version](https://img.shields.io/pypi/v/bulldozer-dtm?color=%2334D058&label=pypi%20package)](https://pypi.org/project/bulldozer-dtm/)
</div>


# Overview
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/result_overview.gif" alt="demo" width="400"/>
</div>

**Bulldozer** is designed as a pipeline that aims to extract a *Digital Terrain Model* (DTM) from a *Digital Surface Model* (DSM). It supports both noisy satellite DSM and high-quality LiDAR DSM.  

# Quick Start

## Installation
You can install **Bulldozer** by running the following command:
```sh
pip install bulldozer-dtm
```
Or you can clone the github repository and use the `Makefile`:
```sh
# Clone the project
git clone https://github.com/CNES/bulldozer.git
cd bulldozer/

# Create the virtual environment and install required depencies
make install

# Activate the virtual env
source bulldozer_venv/bin/activate
```
## Run **Bulldozer**

There are different ways to launch **Bulldozer**:

1. Using the CLI *(Command Line Interface)* - Run the folowing command line after updating the parameters `input_dsm.tif` and `output_dir`:
```console
bulldozer -dsm input_dsm.tif -out output_dir
```
*You can also add optional parameters such as `-dhm`, please refer to the  helper (`bulldozer -h`) command to see all the options.*  

✅ Done! Your DTM is available in the `output_dir`.

2. Using the Python API - You can directly provide the input parameters to the `dsm_to_dtm` function:
```python
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm

dsm_to_dtm(dsm_path="input_dsm.tif", output_dir="output_dir")
```
✅ Done! Your DTM is available in the `output_dir`.

3. Using a configuration file (CLI) - Based on provided [configuration file](conf) templates, you can run the following command line:
```console
bulldozer conf/configuration_template.yaml
```
✅ Done! Your DTM is available in the directory defined in the configuration file.
# License

**Bulldozer**  is licensed under Apache License v2.0. Please refer to the [LICENSE](LICENSE) file for more details.

# <a name="Citation"></a>Citation
If you use **Bulldozer** in your research, please cite the following paper:
```text
@article{bulldozer2023,
  title={Bulldozer, a free open source scalable software for DTM extraction},
  author={Dimitri, Lallement and Pierre, Lassalle and Yannick, Ott},
  journal = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume = {XLVIII-4/W7-2023},
  year = {2023},
  pages = {89--94},
  url = {https://isprs-archives.copernicus.org/articles/XLVIII-4-W7-2023/89/2023/},
  doi = {10.5194/isprs-archives-XLVIII-4-W7-2023-89-2023}
}
```
