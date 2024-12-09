# From satellites data to Digital Terrain Model (DTM)
This section illustrates the use of **Bulldozer** from an open data point cloud to the generation of the corresponding DTM.  
_You can find the corresponding tutorial in notebook format [here](../../notebooks/tutorials/From_Satellites_data_to_DTM.ipynb)._
___

## Prerequisites
To run this tutorial, you require [Python](https://www.python.org/downloads/) (version higher than 3.8) and an Internet connection.



## Data downloading

In this tutorial, we're going to directly use a pre-computed raster format Digital Surface Model (DSM). To produce a DSM from a pair of satellite images, you can for example use [CARS](https://github.com/CNES/cars), an open-source pipeline developed at CNES.

>⚠️ Generating a photogrammetric DSM from satellite images requires at least one pair of images with different acquisition angles over the same area. For example, since SENTINEL-2 data are acquired in Nadir (vertical) configuration, it can't be used to generate a stereoscopic DSM. 


First of all, we'll start by creating a space to store the data for this tutorial and its associated virtualenv: 
``` sh
# Create the working directory
mkdir bulldozer_sat_tuto && cd "$_"
# Create the virtual environment
python -m venv bulldozer_venv
source bulldozer_venv/bin/activate
```
We're going to download DSM generated with the CARS. This DSM was generated in the context of a workshop held at the [FOSS4G 2023 conference](https://2023.foss4g.org/). The aim of this [workshop](https://talks.osgeo.org/foss4g-2023-workshop/talk/9BKXGC/) was to present the application of 3D change detection in the case of a natural disaster using 3D open-source CNES tools on [2023 Turkey/Syria Earthquake](https://en.wikipedia.org/wiki/2023_Turkey%E2%80%93Syria_earthquake) data.
```sh
wget https://github.com/cars-cnes/discover-cnes-3d-tools/raw/gh-pages/outputs_turkey/outputs_cars_pre_event/cars_dsm_pre_event.tif
```

✅ Done! Our data is ready to be used with **Bulldozer**.
<div align="left">
    <img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/tutorial_DSM_Nice.png" width=512>
</div>

## DTM extraction
Now that the data is ready, we can install **Bulldozer** with pip (for an alternative installation method, please refer to the [corresponding section](../INSTALLATION.md)):
```sh
pip install bulldozer-dtm
```
In this tutorial we will use the Command Line Interface (CLI) of **Bulldozer** but there are several different ways of running it (for alternative launch method, please refers to the [corresponding section](../RUN_BULLDOZER.md)):
```sh
bulldozer -in input_dsm.tif -out output_dir -dhm true
```
✅ Done! The resulting DTM is available in `output_dir`:

<div align="left">
    <img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/tutorial_result_DTM_Nice.png" width=512>
</div>
We can also observe the Digital Height Model (DHM), which represents the above-ground structures (buildings, vegetation, etc.) and ignores the topography:

<div align="left">
    <img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/tutorial_result_DHM_Nice.png" width=512>
</div>

With tools like [QGIS](https://www.qgis.org/en/site/) we can also draw profile to visualize the DTM _(red line: DTM, black line: DSM)_:

<div align="left">
    <img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/tutorial_result_profile_Turkey.png" width=512>
    
</div>