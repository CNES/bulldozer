# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).


---

## [Unreleased]


---

## [1.3.0] - 2026-04-07

### Added
- Added `NOTICE` file
- Added regular mask in output mask folder
- Added optional postprocessing step ensuring DTM is below DSM with the parameter `-below_dsm`
- Added documentation

### Changed
- Replaced `isort`, `flake8 plugins`, `flake8`, `pylint` by [`Ruff`](http://docs.astral.sh/ruff/)
- Switched to `pyproject.toml` packaging (removed `setup.cfg`) for PEP 621 and modern setuptools compliance
- Added `VERSION` file (required by the new `pyproject.toml`)
- Updated minimum supported Python version to **>= 3.10** (previously >= 3.8)
- Updated `mypy.ini`, `MANIFEST.in`, `AUTHORS.md`, `pyproject.toml` and `.gitignore` files
- Added missing `.pyd`, `.dll` in `.gitignore`
- Refactored **Bulldozer** Python API (import)
- Added new sections in `README.md`
- Reduced memory usage by avoiding duplication and memory cache in workers
- Refactored eomultiprocessing module for better MP context management and readability
- Rename option `-dhm` by `-ndsm` and associated result product

### Fixed
- Cleaned changelog and switched to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format
- Ignored package initialization files in `.coveragerc` file
- Fixed neighborhood for right and left columns in regular detection step

---

## [1.2.0] - 2026-01-26

### Added
- Added `method` parameter to control whether intermediate data are kept in memory (`mem`) or written to disk
- Added expert parameter `mp_context` to define the multiprocessing context (`spawn`, `fork`, or `forkserver`)

### Changed
- Refactored codebase to improve overall code quality

### Fixed
- Fixed GitHub Action for Docker image publishing
- Removed unsupported `macos-13` build from GitHub workflows

---

## [1.1.2] - 2025-04-30

### Added
- Released the [QGIS plugin](https://github.com/CNES/bulldozer-qgis-plugin)
- Added a new DSM filling method
- Added GitHub Action to publish Docker images to Docker Hub
- Added initial documentation skeleton (work in progress)

### Changed
- Allowed `reg_filtering_iter` to be set to `0` to disable regular mask filtering
- Removed `cloth_tension_force` parameter
- EOScale can now handle custom tile sizes
- Increased the number of regular mask filtering iterations
- Renamed `dev_mode` alias to `dev` and `unhook_iter` alias to `unhook_it`

### Fixed
- Fixed remaining nodata values after DSM filling
- Fixed `get_max_pyramid_level` returning negative values
- Updated log file naming to comply with Windows filename rules
- Fixed nodata handling in intermediate profiles when using `dev` mode during DSM filling

---

## [1.1.1] - 2025-03-25

### Added
- Added Dockerfile and published image on [Docker Hub](https://hub.docker.com/r/cnes/bulldozer)
- Added expert mode to the CLI
- Added deployment pipeline in GitLab CI
- Added `reg_filtering_iter` parameter in expert mode

### Changed
- Refactored C++ sources
- Allowed CLI parameters to override the configuration file
- Added a developer-mode step to write the regular mask before filtering
- Added new border nodata mask computation
- Included border nodata mask option in filling methods

### Fixed
- Fixed GitHub Action for publishing wheels to PyPI
- Refactored configuration file templates
- Removed GitPython dependency (rarely used)
- Added default value `2` for `dezoom_factor` when provided value is less than 2
- Added default value `1` for `nb_iterations` when provided value is less than 1

---

## [1.1.0] - 2025-02-19

### Added
- Built and published wheels for Windows, macOS, and Ubuntu on PyPI
- Filtered small elements in regular masks to reduce noise-related clipping
- Added `--long-usage` CLI option
- Added raster profile information to logs

### Changed
- Introduced new version management strategy
- Updated CLI helper

### Fixed
- Reduced DSM filling runtime with a new filling method
- Fixed regular area detection issues at DSM edges

---

## [1.0.2] - 2024-11-20

### Added
- Major method improvement (pseudo-local approach)
- Added optional ground pre-detection (`ground_clipping`)
- Added optional ground mask input
- Updated parallel computation method (EOScale)
- Added CLI options alongside configuration-file-based CLI
- Added Makefile
- Added new nodata filling method
- Added warnings for unused configuration parameters
- Added project logo

### Changed
- Renamed multiple parameters
- Split quality mask into separate output masks
- Performed global refactoring
- Updated documentation (README, CONTRIBUTING, etc.)
- Updated version storage location
- Updated pits filling method
- Updated log file naming
- Removed `nodata` and `min_valid_height` parameters (now documented via raster metadata)
- Renamed `max_object_width` to `max_object_size`
- Removed deprecated parameters: `keep_inter_dtm`, `four_connexity`, `mp_tile_size`

### Fixed
- Removed problematic `uniform_filter_size` parameter
- Removed `output_resolution` parameter (degraded downsampling)
- Handled missing Git installation gracefully
- Fixed border nodata detection
- Added missing logging level
- Fixed DSM handling with NaN/None nodata values
- Fixed missing DHM nodata values
- Optimized memory release in pipeline
- Fixed CRS handling when `to_authority()` returns `None`
- Added Bulldozer version to log files
- Removed obsolete documentation pending new official docs
- Removed unnecessary memory allocation during anchor prediction

---

## [1.0.1] - 2023-07-05

### Added
- Introduced new API with optional parameters
- Allowed running Bulldozer without a configuration file
- Added default values for all parameters except DSM path and output directory
- Added option to store intermediate DTM

### Changed
- Updated parameter names in the new API
- Improved multiprocessing with better metadata handling
- Updated quality mask format
- Updated unhook method
- Improved overall runtime performance

### Fixed
- Fixed missing DHM when output resolution matched input DSM
- Fixed missing platform information in log files
- Improved nodata retrieval
- Fixed missing disturbed area lines
- Fixed border nodata issues in multiprocessing

---

## [1.0.0] - 2023-03-10

### Added
- Initial open-source release
