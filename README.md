# Bulldozer

## What does it do ?

DTM extraction from DSM (provided by CARS or MicMac for example) using a modified and scalable drape cloth algorithm.

To compute a DTM from a DSM with a size of 40000 x 40000 pixels, it takes 8 minutes approximately using a full node with 24 cores and 60 GBytes of RAM Memory.

## Setup

First, you need to clone Bulldozer in the directory of your choice:

`git clone git@gitlab.cnes.fr:ai4geo/lot6/bulldozer.git`

And go within Bulldozer directory:

`cd bulldozer`

Make sure you have exported $https_proxy and $http_proxy environment variables to install external required dependencies for Bulldozer.

### On AI4GEO VRE

Simply run:

`pip install . --user`

### On HAL

First you need to load python with a version >=3.7, for example:

`module load python/3.8.4`

An then:

`pip install . --user`


## Usage

Modify the file configuration_template.yaml in the conf directory and then run:

`python bulldozer/bulldozer_pipeline.py conf/configuration_template.yaml`

## Contacts

for any help or suggestion, feel free to contact the authors:

- Pierre Lassalle : pierre.lassalle@cnes.fr
- Dimitri Lallement : dimitri.lallement@cnes.fr



## Credits

This package was created with Cookiecutter and the [ai4geo/cookiecutter-python](https://gitlab.cnes.fr/ai4geo/lot2/cookiecutter-python) project template.

## Contributing

Commit messages follow rules defined by [Conventional Commits](https://www.conventionalcommits.org).  
Documentation : [Google style Python Docstring](https://google.github.io/styleguide/pyguide.html) used with [PEP 484 type annotations](https://www.python.org/dev/peps/pep-0484/).  
Style : [PEP 8](https://www.python.org/dev/peps/pep-0008/#other-recommendations) used.
*Copyright 2021 PIERRE LASSALLE & DIMITRI LALLEMENT  
All rights reserved*
