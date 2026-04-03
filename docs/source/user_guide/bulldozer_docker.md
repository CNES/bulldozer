# Using Bulldozer with Docker

**Bulldozer** is available as a Docker image, allowing users to run the full processing pipeline without installing Python dependencies locally.

You can use it either with a [configuration file](#usage-with-a-configuration-file) or with [direct parameters](#usage-with-direct-parameters).  

This approach is particularly useful for:
- Running Bulldozer without Python installation
- Cloud or containerized workflows
- HPC environments supporting containers (Docker / Apptainer)

---

## Pull the Docker image

The official **Bulldozer** image is hosted on [Docker Hub](https://hub.docker.com/r/cnes/bulldozer).

Download the image with:

```sh
docker pull cnes/bulldozer:latest
```

This image contains:
- Bulldozer
- All required Python dependencies
- The full CLI environment

---

## Usage with a configuration file

Bulldozer is executed inside the container while mounting a local directory containing the input data.

The directory is mounted inside the container as `/data`.

---

### Linux / macOS

```sh
docker run --user $(id -u):$(id -g) -v <absolute/path>:/data cnes/bulldozer:latest /data/<config>.yaml
```

---

### Windows (PowerShell)

```powershell
docker run -v C:<absolute/path>:/data cnes/bulldozer:latest /data/<config>.yaml
```

!!! info
    The `--user` option is required on Linux/macOS to avoid file permission issues.
    It is not needed on Windows due to the way Docker Desktop handles filesystem permissions.

---

### Important notes

You must replace:
- `<absolute/path>` with the absolute path to a directory on your system
- `<config>.yaml` with your **Bulldozer** configuration file

Example directory structure:
```
project/
│
├── input_dsm.tif
├── config.yaml
└── output/
```

The container will see this directory as:
```
/data/
```

---

### Configuration file requirement

The configuration file must reference paths inside the container.

Example configuration:
```yaml
dsm_path: "/data/input_dsm.tif"
output_dir: "/data/output"
```

!!! warning
    The output directory **must also be located inside `/data`** so that results are written back to the host system.

---

### Example workflow

1. Prepare a working directory:
```
my_project/
├── input_dsm.tif
├── config.yaml
└── results/
```

2. Example configuration file:
```yaml
dsm_path: "/data/input_dsm.tif"
output_dir: "/data/results"
generate_ndsm: true
nb_max_workers: 8
```

3. Run Bulldozer:
```sh
docker run --user $(id -u):$(id -g) -v $(pwd)/my_project:/data cnes/bulldozer:latest /data/config.yaml
```

Results will appear in:
```
my_project/results/
```

---

## Usage with direct parameters

Instead of using a configuration file, you can also pass CLI arguments directly.

Example:
```sh
docker run --user $(id -u):$(id -g) -v $(pwd):/data cnes/bulldozer:latest -dsm /data/input_dsm.tif -out /data/results -ndsm -workers 8
```

All CLI parameters available in the [**Bulldozer** command-line interface](bulldozer_cli.md) are supported.

---

## Output products

The results products are listed in the [**Bulldozer** outputs](bulldozer_outputs.md) page.

---

## Performance considerations

When running Bulldozer in Docker:
- Use `-workers` to control CPU usage.
- Ensure sufficient RAM is available for large DSMs.
- Use `-mtd mem` for faster processing on systems with sufficient RAM.
- Mount input/output directories on fast storage.

