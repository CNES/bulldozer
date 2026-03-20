# Bulldozer options
This page provides a comprehensive reference of all available parameters in **Bulldozer**.  
These options allow users to control the behavior of the processing pipeline, adapt the tool to different types of input data, and fine-tune the DTM extraction results.

## Options
These options cover essential inputs such as the DSM file, output directory, and common processing settings.  
They are designed to be simple and sufficient for most use cases, allowing users to run the tool efficiently without advanced configuration.


| Name                     | Description                                                                 | Type    | Default Value | Required |
|--------------------------|-----------------------------------------------------------------------------|---------|---------------|----------|
| `generate_ndsm`          | If True, generates the Digital Height Model (DHM = DSM - DTM)              | boolean | False          | No       |
| `max_object_size`        | Maximum size of foreground objects (in meters)                             | float | 16            | No       |
| `ground_mask_path`       | Path to the binary ground classification mask                              | string  | None          | No       |
| `activate_ground_anchors`| Enable ground anchor detection (ground pre-detection)                      | boolean | False         | No       |
| `nb_max_workers`         | Maximum number of CPU cores to use                                         | integer | None          | No       |
| `developer_mode`         | Keep intermediate results                                                  | boolean | False         | No       |


## Expert options
The **Expert options** section provides advanced parameters for experienced users who need finer control over the algorithm.  
These settings enable detailed tuning of the internal processing steps, such as filtering, object detection, and interpolation strategies.  
They should be used with care, as improper configuration may impact the quality of the generated DTM.

!!! warning
    These parameters are intended for advanced users. It is recommended to keep the default values.

| Name                   | Description                                                                 | Type    | Default Value                  | Required |
|------------------------|-----------------------------------------------------------------------------|---------|--------------------------------|----------|
| `reg_filtering_iter`   | Number of regular mask filtering iterations                                 | integer | None (auto: max_object_size/4) | No       |
| `dsm_z_accuracy`       | Altimetric height accuracy of the input DSM (meters)                        | float   | None (auto: 2× resolution)     | No       |
| `max_ground_slope`     | Maximum slope of the terrain (in %)                                         | float   | 20.0                           | No       |
| `prevent_unhook_iter`  | Number of unhook iterations                                                 | integer | 10                             | No       |
| `num_outer_iter`       | Number of gravity step iterations                                           | integer | 25                             | No       |
| `num_inner_iter`       | Number of tension iterations                                                | integer | 5                              | No       |
| `mp_context`           | Multiprocessing context (spawn, fork, forkserver)                           | string  | "fork"                         | No       |
| `intermediate_write`   | Write intermediate results to disk instead of keeping them in memory        | boolean | False                          | No       |