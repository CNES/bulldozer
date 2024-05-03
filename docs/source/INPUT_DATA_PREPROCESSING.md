# Input data preprocessing

This section describes a list of pre-processing steps that can be applied to **Bulldozer** input data to produce a DTM that meets user requirements:

* [Handle Point Cloud (PC) format DSM](#handle-point-cloud-pc-format-dsm)
* [Manage the nodata value](#manage-the-nodata-value)
* [Update DSM and DTM resolution](#update-dsm-and-dtm-resolution)
* [Change the DSM and DTM Coordinates Reference System (CRS)](#change-the-dsm-and-dtm-coordinates-reference-system-crs)

___


## Handle Point Cloud (PC) format DSM
If you're working with DSM in point cloud format, you'll have to convert it to raster format before using it with **Bulldozer**.  
To do this, you can use [cars-rasterize](https://github.com/CNES/cars-rasterize) tool with the following command:
```sh
las2tif dsm.laz raster_format_dsm.tif
```

## Manage the nodata value
To check the nodata value of your input DSM, you can use the [gdalinfo](https://gdal.org/programs/gdalinfo.html) command in the GDAL library:
```sh
gdalinfo input_dsm.tif | grep NoData
```
If you're using a nodata value that doesn't appear in the **Bulldozer** input DSM metadata, you have to update the metadata.    
To do this, you can use the [rasterio](https://rasterio.readthedocs.io/en/stable/) library. For example, if you want to set the nodata value to `-32768`, you can do so using the following Python script:
```python
import rasterio

input_dsm = '/home/il/user/input_dsm.tif'

with rasterio.open(input_dsm, 'r+') as dsm_dataset:
    dsm_dataset.nodata = -32768    
    dsm_dataset.close()
```

## Update DSM and DTM resolution

If you wish to modify the resolution of the DTM produced with **Bullodzer**, you must update the resolution of the input DSM.  
To resample a DSM, you can use the command [gdalwarp](https://gdal.org/programs/gdalwarp.html) from the GDAL library.  
For example, if you want to produce a 30m DSM from a 50cm DSM, you can use the following command:
```sh
gdalwarp -tr 30 30 -r cubic input_DSM.tif downsampled_DSM.tif
```
You can then provide this 30m resolution DSM as input to **Bulldozer** in order to generate a 30m resolution DTM.

## Change the DSM and DTM Coordinates Reference System (CRS)
If you wish to modify the CRS of the DTM produced with **Bullodzer**, you must update the CRS of the input DSM.  
To reproject a DSM, you can use the command [gdalwarp](https://gdal.org/programs/gdalwarp.html) from the GDAL library.  
For example, if you want to reproject a DSM frm the `RGF93 v1 / Lambert-93` (ESPG:2154) CRS into the `WGS 84 / UTM zone 32N` (EPSG:32632) datum, you can use the following command:
```sh
gdalwarp -t_srs EPSG:32632 input.tif output.tif
```
You can then provide this reprojected DSM as input to **Bulldozer** in order to generate a DTM in the CRS you want.

