# Bulldozer, a DTM extraction tool requiring only a DSM as input


<div align="center">
<img src="images/result_overview.gif" alt="demo" width="400"/>
</div>

**Bulldozer** is a free scalable open source Digital Terrain Model (DTM) extraction tool developed at [CNES](https://cnes.fr/fr/).  
This software supports both noisy satellite Digital Surface Models (DSM) and high-quality Lidar DSM.  
It's designed as a pipeline of standalone and exchangeable functions, which combined produce a DTM.

You can install **Bulldozer** by running the following command:
```sh
pip install bulldozer-dtm
```
It has been developed with the goal to provide a simple user interface. For example, to launch **Bulldozer** without any specific configuration, you can use the following command line:
```console
bulldozer -in input_dsm.tif -out output_dir
```
Or use it through the Python API:
 ```python
   from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
   # Example with a specific number of workers
   dsm_to_dtm(dsm_path="input_dsm.tif", output_dir="output_dir")
   ```

