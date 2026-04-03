<div align="center">
    <img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/logo_with_text.png" width=600>

**Bulldozer, a DTM extraction tool that requires only a DSM as input.**

[![pypi](https://img.shields.io/pypi/v/bulldozer-dtm?color=%2334D058&label=pypi)](https://pypi.org/project/bulldozer-dtm/)
[![docker](https://badgen.net/docker/size/cnes/bulldozer?icon=docker&label=image%20size)](https://hub.docker.com/r/cnes/bulldozer)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-blue)
[![Documentation](https://readthedocs.org/projects/bulldozer/badge/?version=stable)](https://bulldozer.readthedocs.io/?badge=stable)</div>


# 🌏Overview
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/result_overview.gif" alt="demo" width="400"/>
</div>
 
**Bulldozer** is a pipeline designed to extract a *Digital Terrain Model* (DTM) from a *Digital Surface Model* (DSM). It supports both noisy satellite DSMs and high-quality LiDAR DSMs.

# ✨ Features

- Digital Terrain Model (DTM) extraction from a single Digital Surface Model (DSM)
- Requires only a DSM as mandatory input
- Optional ground / non-ground mask support
- Designed for both noisy satellite DSMs and high-quality LiDAR DSMs
- Command-line interface (CLI)
- Python API
- Configuration-file based execution
- Docker container support
- Fully open-source (Apache 2.0)

# 🚀 Quick Start

## 🛠️ Installation

You can install **Bulldozer** by running the following command:
```sh
pip install bulldozer-dtm
```
*For alternative installation methods, please refer to the [documentation](https://bulldozer.readthedocs.io/en/stable/installation.html).*

### Requirements
- Python ≥ 3.10

## ⚙️ Run **Bulldozer**

As described in the [documentation](https://bulldozer.readthedocs.io/en/stable/run_bulldozer.html), there are many different ways to launch **Bulldozer**. Here are the three most popular:

1. Using the CLI *(Command Line Interface)* - Run the following command line after updating the parameters `input_dsm.tif` and `output_dir`:
```console
bulldozer -dsm input_dsm.tif -out output_dir
```
*You can also add optional parameters such as `-ndsm`, please refer to the  helper (`bulldozer -h`) command to see all the options.*  

✅ Done! Your DTM is available in the `output_dir`.

2. Using the Python API - You can directly provide the input parameters to the `dsm_to_dtm` function:
```python
from bulldozer import dsm_to_dtm

dsm_to_dtm(dsm_path="input_dsm.tif", output_dir="output_dir")
```
✅ Done! Your DTM is available in the `output_dir`.

3. Using a configuration file (CLI) - Based on provided [configuration file](https://github.com/CNES/bulldozer/tree/master/conf) templates, you can run the following command line:
```console
bulldozer conf/configuration_template.yaml
```
✅ Done! Your DTM is available in the directory defined in the configuration file.

## 🐋 **Bulldozer** docker image

**Bulldozer** is available on Docker Hub and can be downloaded using:
``` bash
docker pull cnes/bulldozer
```

And you can run **Bulldozer** with the following command:

**Linux / macOS**
``` bash
docker run --user $(id -u):$(id -g) -v <absolute/path>:/data cnes/bulldozer:latest /data/<conf>.yaml
```
**Windows (PowerShell)**
``` powershell
docker run --rm  -v C:<absolute/path>:/data cnes/bulldozer:latest /data/<conf>.yaml
```

⚠️ You have to change the `<absolute/path>` to provide a valid absolute path to a directory where the input data are stored and where **Bulldozer** will write the output DTM. You also have to provide a configuration file (and rename `<conf>.yaml` in the command line) in this directory with an `output_dir` value using the `data` folder given to docker, e.g.: `output_dir : "/data/outputdir"`.  

# ✒️ Credits

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
# 📜 License

**Bulldozer** is licensed under permissive [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Please refer to the [LICENSE](LICENSE) file for more details.


# 🆘 Support

For issues, questions, or feature requests, please open an issue on our [GitHub Issues page](https://github.com/CNES/bulldozer/issues) or check the documentation for additional resources.


# 🤝Contributing
We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get involved, including coding standards and submission processes.


