# Bulldozer, a DTM extraction tool requiring only a DSM as input

<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/result_overview.gif" alt="demo" width="400"/>
</div>

**Bulldozer** is a free scalable open source Digital Terrain Model (DTM) extraction tool developed at [CNES](https://cnes.fr/en/).  
This software supports both noisy satellite Digital Surface Models (DSM) and high-quality Lidar DSM.  

**Bulldozer** is part of the 3D image processing chain of the [CO3D satellite mission](https://cnes.fr/projets/co3d).  
It's also used in [Evoland](https://www.evo-land.eu/) project.

## Key Features

- Digital Terrain Model (DTM) extraction from a single Digital Surface Model (DSM)
- Requires only a DSM as mandatory input
- Optional ground / non-ground mask support
- Designed for both noisy satellite DSMs and high-quality LiDAR DSMs
- Command-line interface (CLI)
- Python API
- Configuration-file based execution
- Docker container support
- Fully open-source (Apache 2.0)