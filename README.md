<div align="center">
    <img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/logo_with_text.png" width=600>

**Bulldozer, a DTM extraction tool that requires only a DSM as input.**

[![pypi](https://img.shields.io/pypi/v/bulldozer-dtm?color=%2334D058&label=pypi)](https://pypi.org/project/bulldozer-dtm/)
[![docker](https://badgen.net/docker/size/cnes/bulldozer?icon=docker&label=image%20size)](https://hub.docker.com/r/cnes/bulldozer)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
</div>


# Overview
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/result_overview.gif" alt="demo" width="400"/>
</div>
 
**Bulldozer** is a pipeline designed to extract a *Digital Terrain Model* (DTM) from a *Digital Surface Model* (DSM). It supports both noisy satellite DSMs and high-quality LiDAR DSMs.

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

## **Bulldozer** docker image

[![Docker Status](http://dockeri.co/image/cnes/bulldozer)](https://hub.docker.com/r/cnes/bulldozer)

**Bulldozer** is available on Docker Hub and can be downloaded using:
``` bash
docker pull cnes/bulldozer
```
And you can run **Bulldozer** with the following command:
``` bash
docker run --user $(id -u):$(id -g) --shm-size=10gb -v <path>:/data cnes/bulldozer:latest /data/<conf>.yaml
```
⚠️ You have to change the `<path>` to provide a valide absolute path to a directory where the input data are stored and where **Bulldozer** will write the ouput DTM. You also have to provide a configuration file (and rename `<conf>.yaml` in the command line) in this directory with an `ouput_dir` value using the `data` folder given to docker, e.g.: `output_dir : "/data/outputdir"`. If you want to run **Bulldozer** on a huge DSM, please improve the shared memory value of the command line (`--shm-size`) argument.  



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
