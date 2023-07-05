<div align="center">
    <img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/bulldozer_logo.png" width=256>

**Bulldozer, a DTM extraction tool requiring only a DSM as input.**

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Version](https://img.shields.io/pypi/v/bulldozer-dtm?color=%2334D058&label=pypi%20package)](https://pypi.org/project/bulldozer-dtm/)


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
<img src="docs/source/images/result_overview.gif" alt="demo" width="400"/>
</div>

**Bulldozer** is designed as a pipeline of standalone functions that aims to extract a *Digital Terrain Model* (DTM) from a *Digital Surface Model* (DSM).  
But you can also use one of the following function without running the full pipeline:
* **DSM preprocessing**
    * **Nodata extraction:** a group of methods to differentiate and extract nodata related to failed correlations during the DSM computation and those of the image border
    * **Disturbed areas detection:** a method to locate disturbed areas. These noisy areas are mainly related to areas in which the correlator has incorrectly estimated the elevation (water or shadow areas).
* **DTM extraction**
    * **DTM computation:** the main method that extracts the DTM from the preprocessed DSM.
* **DTM postprocessing**
    * **Pits detection:** a method to detect pits in the provided raster and return the corresponding mask.
    * **Pits filling:** a method to fill pits in the generated DTM (or input raster).
    * **DHM computation:** a method to extract the *Digital Height Model* (DHM).

For more information about these functions and how to call them, please refer to the <a href="#notebooks">notebook documentation section</a>.

## Installation

### Using Pypi
You can install **Bulldozer** by running the following command:
```sh
pip install bulldozer-dtm
```
### Using Github

It is recommended to install **Bulldozer** into a virtual environment, like [conda](https://docs.conda.io/en/latest/) or [virtualenv](https://virtualenv.pypa.io/en/latest/).

* Installation with `virtualenv`:

```sh
# Clone the project
git clone https://github.com/CNES/bulldozer.git
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
dsm_path : "<input_dsm.tif>"
# Output directory path (if the directory doesn't exist, create it)
output_dir : "<output_dir>"
```
2. Run the pipeline:
   * Through CLI *(Command Line Interface)*
   ```console
   bulldozer --conf conf/configuration_template.yaml
   ```
   * Through Python API using the config file
   ```python
   from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm

   dsm_to_dtm(config_path="conf/configuration_template.yaml")
   ```
   * Through Python API providing directly the input parameters (missing parameters will be replaced by default values)
   ```python
   from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
   # Example with a specific number of workers
   dsm_to_dtm(dsm_path="<input_dsm.tif>", output_dir="<output_dir>", nb_max_workers=16)
   ```

3. ✅ Done! Your DTM is available in the *<output_dir>*

## Documentation

### Notebooks

For each section described in <a href="#key-features">Key Features</a> section you can follow one of the tutorial notebook:
* [Running Bulldozer (full pipeline)](docs/notebooks/0_bulldozer_pipeline.ipynb)
* [Preprocessing standalone functions](docs/notebooks/1_preprocess.ipynb)
* [Extraction step](docs/notebooks/2_DTM_extraction.ipynb)
* [Postprocessing standalone functions](docs/notebooks/3_postprocess.ipynb)

### Full documentation (WIP)
**Bulldozer** also has a more detailed documentation.  
A high-level overview of how it’s organized will help you know where to look for certain things:

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

**Bulldozer** has a Apache V2.0 license, as found in the [LICENSE](LICENSE) file.

## Credits

Please refer to the [Authors file](AUTHORS.md).

## Reference

 [D. Lallement, P. Lassalle, Y. Ott, R. Demortier, and J. Delvit, 2022. BULLDOZER: AN AUTOMATIC SELF-DRIVEN LARGE SCALE DTM EXTRACTION METHOD FROM DIGITAL SURFACE MODEL. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B2-2022/409/2022/)