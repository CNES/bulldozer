<div align="center">
    <img src="docs/source/_static/images/bulldozer-logo.svg" width=512>

**Bulldozer, a DTM extraction tool requiring only a DSM as input.**

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://readthedocs.org/projects/cars/badge/?version=latest)](https://github.com/CNES/bulldozer)

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •  
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contribute">Contribute</a> •
  <a href="#licence">Licence</a> •
  <a href="#reference">Reference</a>
</p>

</div>

---

## Key features

<div align="center">
<img src="docs/source/images/overview_dsm_3d.gif" alt="drawing" width="400"/>
</div>


**Bulldozer** is designed as a pipeline of standalone functions.
* **DSM preprocessing**
    * **Nodata extraction:** 
    * **Disturbed areas detection:**
* **DTM extraction**
    * **Nodata extraction:** 

* **DTM postprocessing**
    * **Nodata extraction:** 

## Installation

It is recommended to install **Bullodzer** into a virtual environment, like `conda` or `virtualenv`.

* Installation with `virtualenv`:

```sh
# Clone the project
git clone https://github.com/CNES/bullodzer.git
cd bulldozer/

# Create the environment
python -m venv bulldozer_venv
source bulldozer_venv/bin/activate

# Install the library
pip install .
```
## Quick Start

1. First you have to create a configuration file or edit the `configuration_template.yaml` available in the `conf` directory. You have to update at least the following parameters:
```yaml
# Input DSM path (expected format: "<folder_1>/<folder_2>/<file>.<[tif/tiff]>")
dsmPath : "<input_dsm.tif>"
# Output directory path (if the directory doesn't exist, create it)
outputDir : "<output_dir>"
```
2. Run the pipeline:

```console
bullodzer --conf conf/configuration_template.yaml
```
3. ✅ Done! Your DTM is available in the *<output_dir>*
## Documentation

**Bullodzer** has some documentation. A high-level overview of how it’s organized will help you know where to look for certain things:

* [Tutorials](docs/tutorials/index.md) take you by the hand through a series of steps to create a DLCooker application. Start here if you’re new to DLCooker.
* [How-to guides](docs/how-to/index.md) are recipes. They guide you through the steps involved in addressing key problems and use-cases. They are more advanced than tutorials and assume some knowledge of how DLCooker works.
* [Explanation guides](docs/explanation/index.md) discuss key topics and concepts at a fairly high level and provide useful background information and explanation.

> **_NOTE:_** The documentation is not available online yet, it needs to be built manually.

To do so, please execute the following command at the root:

```console
mkdocs serve
```

## Contribute

To do a bug report or a contribution, see the [**Contribution Guide**](CONTRIBUTING.md).  
for any help or suggestion, feel free to contact the authors:

- Dimitri Lallement : dimitri.lallement@cnes.fr
- Pierre Lassalle : pierre.lassalle@cnes.fr
## Licence

**Bullodzer** has a Apache V2.0 license, as found in the [LICENSE](LICENSE) file.

## Credits

Please refer to the [Authors file](AUTHORS.md).

## Reference

 [D. Lallement, P. Lassalle, Y. Ott, R. Demortier, and J. Delvit, 2022. BULLDOZER: AN AUTOMATIC SELF-DRIVEN LARGE SCALE DTM EXTRACTION METHOD FROM DIGITAL SURFACE MODEL. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B2-2022/409/2022/)