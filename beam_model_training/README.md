# BEAM_model_traning

The expected folder structure is as follow (set the location of project_directory to 'ROOT_PATH' in my_paths.py):

├── project_directory
│ ├── images // A list of files in GeoTIFF format.
│ │ ├── image.tiff
│ ├── labels // A shapefile or csv file (in Google Open Buildings Dataset format) containing all labels for the included images.
│ │ ├── label.shp
│ ├── models
│ │ ├── saved_models.pkl

# Development environment - installation steps

Run the install.bat file as suggested by the documentation. After the installation has completed, the beam environment should have been created and made available to use.

## VS Code

In order to use run VS Code within the Mamba environment, you can follow the instructions provided in the [Usage section](https://github.com/conda-forge/miniforge#usage) of miniforge.
