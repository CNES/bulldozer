# Changelog

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

## 1.0.0 Open Source Release (January 2023)
### Added

- Publication of the code in open-source