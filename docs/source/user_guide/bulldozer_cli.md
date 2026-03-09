# Bulldozer Command-Line Interface (CLI)

**Bulldozer** provides a command-line interface (CLI) to extract a Digital Terrain Model (DTM) from a Digital Surface Model (DSM).

The CLI can be used in two main ways:
1. [Using a configuration file](#usage-with-a-configuration-file)
2. [Passing parameters directly via command-line arguments](#usage-with-command-line-parameters)

---

## General help

To display the short help message:
```sh
bulldozer --help
```

To display extended help:
```sh
bulldozer --long_usage
```

To display expert parameters:
```sh
bulldozer --expert_mode
```

To show the installed version:
```sh
bulldozer --version
```

---

## Usage with a configuration file

```sh
bulldozer config.yaml
```

This mode allows you to define all parameters in a configuration file.

If additional CLI arguments are provided, they override the values defined in the configuration file.

ℹ️Configuration templates are available in the `conf` directory. You may also create your own configuration files, provided that they include at least the two mandatory parameters: `dsm_path` and `output_dir`.

The available optional parameters correspond to the long-form command-line options (`--<parameter>`) listed in the section [usage with command line parameters](#usage-with-command-line-parameters). For a detailed description of their behavior, please refer to the page: [**Bulldozer** parameters](https://bulldozer.readthedocs.io/en/stable/bulldozer_parameters.html).

ℹ️This mode is recommended for:
- Reproducible workflows
- Large-scale production processing
- Batch execution on HPC systems

---

## Usage with command-line parameters

### Minimal example

```sh
bulldozer -dsm input_dsm.tif -out output_directory
```

### Required arguments

| Argument               | Description                      |
|------------------------|----------------------------------|
| `-dsm`, `--dsm_path`   | Path to the input DSM (GeoTIFF). |
| `-out`, `--output_dir` | Output directory.                |

---

### Optional parameters (standard mode)

| Argument                                              | Description                                        |
|-------------------------------------------------------|----------------------------------------------------|
| `-ndsm`, `--generate_ndsm`                            | Generate a Digital Height Model (DHM = DSM − DTM). |
| `-max_size <value>`, `--max_object_size <value>`      | Maximum size of foreground objects (meters).       |
| `-ground <mask.tif>`, `--ground_mask_path <mask.tif>` | Path to a binary ground classification mask.       |
| `-anchors`, `--activate_ground_anchors`               | Activate automatic ground anchor detection.        |
| `-workers <value>`, `--nb_max_workers <value>`        | Maximum number of CPU cores to use.                |
| `-dev`, `--developer_mode`                            | Keep intermediate processing results.              |

---

### Expert parameters

These optional parameters control internal algorithm behaviour and should be modified only by advanced users.

| Argument                                              | Description                                                                           |
|-------------------------------------------------------|---------------------------------------------------------------------------------------|
| `-mtd <value>`, `--method <value>`                    | Processing mode: `mem` (all in memory) or `no-mem` (write temporary results to disk). |
| `-reg_it <value>`, `--reg_filtering_iter <value>`     | Number of regular mask filtering iterations.                                          |
| `-dsm_z <value>`, `--dsm_z_accuracy <value>`          | Altimetric accuracy of the input DSM (meters). Default: 2 × DSM resolution.           |
| `-max_slope <value>`, `--max_ground_slope <value>`    | Maximum expected terrain slope (%).                                                   |
| `-unhook_it <value>`, `--prevent_unhook_iter <value>` | Number of unhook iterations.                                                          |
| `-outer <value>`, `--num_outer_iter <value>`          | Number of gravity step iterations.                                                    |
| `-inner <value>`, `--num_inner_iter <value>`          | Number of tension iterations.                                                         |
| `-context <value>`, `--mp_context <value>`            | Multiprocessing context (`spawn`, `fork`, or `forkserver`).                           |

### Command with options example

```sh
bulldozer -dsm input_dsm.tif -out results -ndsm -workers 8 -max_size 32 -mtd mem
```

---

## Output products

The results products are listed in the [**Bulldozer** ouputs](https://bulldozer.readthedocs.io/en/stable/bulldozer_outputs.html) page.

---

## Performance considerations

- Use `-workers` to control CPU usage.
- Use `-mtd mem` for faster processing on systems with sufficient RAM.
- Use `-mtd no-mem` for large DSMs or memory-constrained environments.
- On HPC systems,  ensure allocated CPU cores match the `-workers` value.
