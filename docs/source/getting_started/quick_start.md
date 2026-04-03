# Quick Start

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

If you’d like to go further: explore the different ways to [run **Bulldozer**](../user_guide/how_to_run_bulldozer.md), look into the various available [options](../user_guide/bulldozer_options.md), check out **Bulldozer**’s [outputs](../user_guide/bulldozer_outputs.md), or learn how to use the tool through a [tutorial](../tutorials/from_lidar_to_dtm.md).
