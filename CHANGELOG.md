# Changelog

## 1.0.2 New pipeline (November 2024)

### Added
- Update of the version release policy: increased release frequency to be expected
- Major method improvment (pseudo-local)
- Adding anchor points detection (optional)
- Adding ground mask entry (optional)
- Parallel computation method updated (EOScale)
- Adding CLI with options (in addition to CLI with config file)
- Adding Makefile
- New nodata filling method
- Warning on unused parameters in the configuration

### Changed
- Parameters name change
- Quality mask split into different masks available in the output directory
- Global refactoring
- Documentation update (README, CONTRIBUTING, etc.)
- Update the version location
- Update the pits filling method

### Fixed
- Remove problematic `uniform_filter_size` parameter 
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

## 1.0.0 Open Source Release (January 2023)

### Added
- Publication of the code in open-source