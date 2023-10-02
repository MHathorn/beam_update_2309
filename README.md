# BEAM Readme

Repo for the UNITAC BEAM tool. 

## Installation and Use

Installation instructions to use the tool locally can be found in 'documentation'. 

By default the installation will use the strict environment with pinned version numbers. A minimal solvable environment can be found in environment_solvable in the unitac-backend directory. Training notebooks and scripts can be run in the BEAM environment. 

Model weights can be found at [LINK].

Packaged dependencies (for offline installation) can be found at [LINK].

## Training

Training scripts are in the 'training' directory. Update the paths in utils/my_paths.py to your image locations before running. 

Recommended folder structure (set the location of project_directory to 'ROOT_PATH' in my_paths.py): 

├── project_directory
│   ├── images
│   │   ├── image.tiff
│   ├── labels
│   │   ├── label.shp
│   ├── models
│   │   ├── saved_models.pkl

Full documentation and instructions for training can be found by following the notebooks in the 'notebooks' directory. 