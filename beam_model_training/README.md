# BEAM_model_traning

# Development environment - installation steps 

## BEAM environment

### Windows

1. Clone this repository to your Windows machine.
2. Run the install.bat file as suggested by the documentation. After the installation has completed, the beam environment should have been created and made available to use.

### Linux (Azure VM Data Science)

1. Clone this repository to your Linux machine.
2. Install Mamba on your machine by following the [installation steps](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).
3. Change the current directory in your terminal to your beam_update_2039/ directory, and run the following command to create the BEAM environment.

`mamba env create -f unitac-backend/environment_solvable.yml`

In case of any permissions issues, run through the steps as `sudo` and consult your network administrator to grant r/w permissions for your current user to interact with the Conda environment, data drive and code repository.

## VS Code

In order to use run VS Code within the Mamba environment, you can follow the instructions provided in the [Usage section](https://github.com/conda-forge/miniforge#usage) of miniforge.

## PYTHONPATH

For the modules in `beam_model_training` to be successfully retrieved by the scripts, this directory must be added to the PYTONPATH environment variable. To fix this permanently for your user, follow these steps:

1. Identiy the site packages directory with command:

```bash
python -m site --user-site
```

2. Create directory if it doesn't exist, from git bash, and create file `pythonpath.pth` in the directory:

```bash
USER_SITE_PATH=$(python -m site --user-site)
mkdir -p $USER_SITE_PATH
touch $USER_SITE_PATH/pythonpath.pth
```

3. Open the file as with your prefered code editor and add the path to the `beam_model_training` directory into this file.

#

The expected folder structure at the start of a project is as follow:

├── project_directory  
│ ├── images/ # A list of files in GeoTIFF format.  
│ │ ├── image.TIFF  
│ ├── labels/ # All labels for the included images in shapefile, csv or csv.gz format.  
│ │ ├── label.shp  
| ├── project_config.yaml # The project's configuration file.