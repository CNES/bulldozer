# Bulldozer, a DTM extraction tool requiring only a DSM as input

<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/result_overview.gif" alt="demo" width="400"/>
</div>

**Bulldozer** is a free scalable open source Digital Terrain Model (DTM) extraction tool developed at [CNES](https://cnes.fr/en/).  
This software supports both noisy satellite Digital Surface Models (DSM) and high-quality Lidar DSM.  

## Installation

You can install **Bulldozer** by running the following command:

```sh
pip install bulldozer-dtm
```

*For alternative installation method, please refer to the [corresponding section](getting_started/installation.md).*

## Usage

As described in the [corresponding section](user_guide/how_to_run_bulldozer.md), there are many different ways to launch **Bulldozer**. Here are the two most popular:

### 1. Using the CLI *(Command Line Interface)*
Run the folowing command after updating the parameters `input_dsm.tif` and `output_dir`:

```bash
bulldozer -in input_dsm.tif -out output_dir
```

You can also add optional parameters such as `--ndsm true` (please refer to the [CLI usage section](user_guide/bulldozer_cli.md) to see all the parameters).  
✅ Done! Your DTM is available in the `output_dir`.

### 2. Using the Python API

You can directly provide the input parameters:

```python
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm

dsm_to_dtm(dsm_path="input_dsm.tif", output_dir="output_dir")
```

✅ Done! Your DTM is available in the `output_dir`.