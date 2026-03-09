# Bulldozer Python API

In addition to the command-line interface, **Bulldozer** provides a Python API to extract a Digital Terrain Model (DTM) from a Digital Surface Model (DSM).  

The main entry point of the API is:
```python
from bulldozer import dsm_to_dtm
```


You can use it either with a [configuration file](#usage-with-a-configuration-file) or with [direct parameters](#usage-with-direct-parameters).  

The API is recommended for:
- Integration into scientific workflows
- Jupyter notebook usage
- Automated pipelines
- Custom processing chains

---

## Usage with a configuration file

```python
from bulldozer import dsm_to_dtm

dsm_to_dtm(config_path="conf.yaml")
```

This mode provides:
- Full reproducibility
- Cleaner experiment tracking
- Easier parameter tuning

ℹ️If additional keyword arguments are provided, they override the values defined in the configuration file.

---

## Usage with direct parameters

```python
from bulldozer import dsm_to_dtm

dsm_to_dtm(
    dsm_path="input_dsm.tif",
    output_dir="output_dir"
)
```

### Required parameters

| Parameter    | Description                              | Type    | Default value | Required | 
|--------------|------------------------------------------|---------|---------------|----------|
| `dsm_path`   | Path to the input DSM (GeoTIFF).         | str     | None          | Yes      |
| `output_dir` | Directory where results will be written. | str     | None          | Yes      |

---

### Optional parameters

All parameters available in the CLI are also available in the Python API as keyword arguments.

| Argument                  | Description                                                     | Type    | Default value | Required |
|---------------------------|-----------------------------------------------------------------|---------|---------------|----------|
| `generate_ndsm`           | Generate a Normalized Digital Surface Model (nDSM = DSM − DTM). | bool    | False         | No       |
| `max_object_size`         | Maximum size of foreground objects (meters).                    | float   | 16            | No       |
| `ground_mask_path`        | Path to a binary ground classification mask.                    | str     | None          | No       |
| `activate_ground_anchors` | Activate automatic ground anchor detection.                     | bool    | False         | No       |
| `nb_max_workers`          | Maximum number of CPU cores to use.                             | int     | None (\*)     | No       |
| `developer_mode`          | Keep intermediate processing results.                           | bool    | False         | No       |
(\*) if None, use maximum number of available CPU core. 

---

### Expert parameters

⚠️These parameters control internal algorithm behaviour and should be modified only by advanced users.

| Argument              | Description                                                                               | Type    | Default value | Required |
|-----------------------|-------------------------------------------------------------------------------------------|---------|---------------|----------|
| `method`              | Processing mode: "mem" (all in memory) or "no-mem" (write temporary results to disk). | str     | "mem"       | No       |
| `reg_filtering_iter`  | Number of regular mask filtering iterations.                                              | int     | None (*)      | No       |
| `dsm_z_accuracy`      | Altimetric accuracy of the input DSM (meters). Default: 2 × DSM resolution.               | float   | None (**)     | No       |
| `max_ground_slope`    | Maximum expected terrain slope (%).                                                       | float   | 20.0          | No       |
| `prevent_unhook_iter` | Number of unhook iterations.                                                              | int     | 10            | No       |
| `num_outer_iter`      | Number of gravity step iterations.                                                        | int     | 25            | No       | 
| `num_inner_iter`      | Number of tension iterations.                                                             | int     | 5             | No       | 
| `mp_context`          | Multiprocessing context ("spawn", "fork", or "forkserver").                         | str     | "fork"      | No       |

(\*) if None, use the default value: `int(max_object_size / 4)`.  
(\*\*) if None, use the default value: `2 * planimetric_resolution`.

---

### Example of Python usage with options

```python
from bulldozer import dsm_to_dtm

# Run Bulldozer processing
dsm_to_dtm(
    dsm_path="input_dsm.tif",
    output_dir="results/",
    generate_ndsm=True,
    max_object_size=32,
    nb_max_workers=8,
    method="mem"
)
```

---

## Output products

The function does not return the DTM as a NumPy array.  
It writes results directly to disk for scalability and memory efficiency.  
The results products are listed in the [**Bulldozer** ouputs](https://bulldozer.readthedocs.io/en/stable/bulldozer_outputs.html) page.

---

## Performance considerations

- Use `nb_max_workers` to control CPU usage.
- Use `method="mem"` for faster processing on systems with sufficient RAM.
- Use `method="no-mem"` for large DSMs or memory-constrained environments.
- On HPC systems, ensure allocated CPU cores match the `nb_max_workers` value.

