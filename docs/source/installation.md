# Installation

## Requirements

- **Python ≥ 3.10**
- Recommended: virtual environment

Bulldozer depends on scientific Python libraries such as `numpy`, `scipy`, and `rasterio`.  
On some systems (especially clusters), `rasterio` may require system-level GDAL libraries.

---

## Install from PyPI (recommended)

The simplest way to install **Bulldozer** is via pip:
```sh
python -m venv venv
source venv/bin/activate
pip install bulldozer-dtm
```
Then verify installation:
```sh
bulldozer --help
```

---

## Install from source (GitHub)

Repository:
https://github.com/CNES/bulldozer

### Using the provided Makefile (recommended for developers)
```sh
git clone https://github.com/CNES/bulldozer.git
cd bulldozer

make install
source bulldozer_venv/bin/activate
```
This will:
- Create a dedicated virtual environment (`bulldozer_venv`)
- Install all runtime and development dependencies
- Install pre-commit hooks

### Manual installation with `venv`

```sh
git clone https://github.com/CNES/bulldozer.git
cd bulldozer

python -m venv bulldozer_venv
source bulldozer_venv/bin/activate
pip install .
```
---

## Development installation

To install Bulldozer in editable mode with development tools:

```sh
pip install -e .[dev]
```

This installs:

- Ruff (lint & format)
- mypy
- pytest
- pre-commit

---

## Using Docker

A Docker image is available on Docker Hub:

```sh
docker pull cnes/bulldozer:latest
```

Example usage:

```sh
docker run --rm -v $(pwd):/data cnes/bulldozer:latest bulldozer --help
```