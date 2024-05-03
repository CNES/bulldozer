# Installation
This section describes the different ways of installing **Bulldozer**.
___

## Using Pypi
You can install **Bulldozer** by running the following command:
```sh
pip install bulldozer-dtm
```
## Using Github
* Installation using `Makefile`:
```sh
# Clone the project
git clone https://github.com/CNES/bulldozer.git
cd bulldozer/

# Create the virtual environment and install required depencies
make install

# Activate the virtual env
source venv/bin/activate
```

* Installation with Python and using a [virtualenv](https://virtualenv.pypa.io/en/latest/) (you can also use  [conda](https://docs.conda.io/en/latest/)):
```sh
# Clone the project
git clone https://github.com/CNES/bulldozer.git
cd bulldozer/

# Create the virtual environment
python -m venv bulldozer_venv
source bulldozer_venv/bin/activate

# Install the required dependencies
pip install .
```
