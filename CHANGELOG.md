# Changelog

## 1.1.2 QGIS plugin release (April 2025)

### Added
- [QGIS plugin](https://github.com/CNES/bulldozer-qgis-plugin) released
- New filling DSM method
- Github action for publishing Docker image on DockerHub

### Changed
- Allows the user to set the `reg_filtering_iter` to 0 in order to desactivate the regular mask fitlering
- Remove `cloth_tension_force` parameter
- Eoscale can now handle specific tile size
- Update the filling DSM method
- Increase the number of filtering iteration of the regular mask
- Rename `dev_mode` alias to `dev` and `unhook_iter` alias to `unhook_it`

### Fixed
- Fix the remaining nodata issue after DSM filling step
- Fix the issue of `get_max_pyramid_level` function returning negative result

---

## 1.1.1 Docker release (March 2025)

### Added
- Adding Dockerfile and publishing it on [dockerhub](https://hub.docker.com/r/cnes/bulldozer)
- Adding expert mode CLI
- Adding deploy pipeline in gitlab CI
- Adding `reg_filtering_iter` parameter in expert mode 

### Changed
- C++ files refactoring
- The user can override the configuration file with parameters in the CLI
- Adding a new step in developer mode to write regular mask before filtering
- New border noadata mask computation
- Including border nodata mask option in filling method

### Fixed
- Github actions workflow file for publishing wheels on PyPI fixed
- Configuration file templates refactoring
- Fixing gitPython issue (dependency removed as it's rarely used)
- Adding default value (2) for dezoom factor if the provided value is less than 2 
- Adding default value (1) for nb_iterations during the regular mask filtering if the provided value is less than 1  

---

## 1.1.0 Cross-platforms compatibility (February 2025)

### Added
- Building and providing wheels for Windows, macOS and Ubuntu on PyPI
- Filter small element in the regular mask to avoid clipping on noisy elevation points
- Adding `--long-usage` option 
- Adding raster profile to logs

### Changed
- New version number management
- Update CLI helper

### Fixed
- Reduce DSM filling runtime by using a new filling method
- Regular areas detection on the DSM edges issue => fixed

---

## 1.0.2 New pipeline (November 2024)

### Added
- Update of the version release policy: increased release frequency to be expected
- Major method improvement (pseudo-local)
- Adding ground pre-detection: ground_clipping (optional)
- Adding ground mask entry (optional)
- Parallel computation method updated (EOScale)
- Adding CLI with options (in addition to CLI with config file)
- Adding Makefile
- New nodata filling method
- Warning on unused parameters in the configuration
- New logo

### Changed
- Parameters name change
- Quality mask split into different masks available in the output directory
- Global refactoring
- Documentation update (README, CONTRIBUTING, etc.)
- Update the version location
- Update the pits filling method
- New log file name
- Remove `nodata` and `min_valid_height` parameters and replace them with an explanation in documentation on how to change the 'nodata' value in raster metadata 
- `max_object_width` parameter is renamed `max_object_size`
- `keep_inter_dtm` parameter is removed due to the new pipeline
- `four_connexity` parameter is removed due to the new pipeline
- `mp_tile_size` parameter is removed due to the new pipeline

### Fixed
- Remove problematic `uniform_filter_size` parameter
- Remove `output_resolution` parameter since it was a degraded downsampling method
- Git exception is handled if the user doesn't have git installer
- Border nodata detection => fixed
- Add missing logging level
- DSM with nodata set to NaN/None value => fixed
- DHM nodata issue (missing value) => fixed
- Optimize memory release in the pipeline
- DSM with CRS for which to_authority() returns None => fixed
- Bulldozer version added in logfile
- Remove unused documentation until the new version of the official documentation is published
- Remove unnecessary memory allocation in anchor prediction

---

## 1.0.1 New interface (July 2023)

### Added
- New API: all parameters are no longer required
- New API: It is possible to launch Bulldozer without a configuration file, simply by supplying the values of the parameters
- Adding default values for input parameters (except DSM path and output directory)
- New option: store intermediate DTM

### Changed
- New API: parameters name update
- Multi-processing improvement (metadata taking in consideration)
- New quality mask format
- Unhook method update
- Global runtime improvement

### Fixed
- DHM was not generated when the output resolution was the same as the input DSM => Fixed
- Logger file did not display the plateform information => Fixed
- Nodata retrieval improved
- Disturbed areas missing line => Fixed
- Border nodata multi-processing issue => Fixed

---

## 1.0.0 Open Source Release (January 2023)

### Added
- Publication of the code in open-source