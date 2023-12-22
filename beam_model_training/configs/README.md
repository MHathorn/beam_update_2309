# Configure the BEAM training tool

## Overview

The configuration file is divided into three main sections:

**General Settings:** Basic configuration for the experiment, including codes and directory paths.

**Data Settings:** Parameters related to the data used in the experiment, such as erosion, labels, and directories.

**Model Settings:** Details about the model architecture, training parameters, and other related settings.

## General Settings

**codes:** A list of strings representing different categories or types in the experiment. For the current scope of the BEAM project this defaults to "Background" and "Building".

**dirs:** A dictionary defining various directory paths used in the experiment. Each key represents a specific type of directory, and its value is the path relative to the root directory. For example, "eval" points to the sub-directory [ROOT_DIR]/eval.

## Data Settings

**erosion:** A boolean value (true or false) indicating whether erosion is applied in the data processing step.

**root_dir:** The root directory for the experiment data, e.g., "F:/ethekwini".

**seed:** An integer value used for random seed setting, ensuring reproducibility.

**test:** A dictionary containing test-specific parameters. Currently, it includes:

**model_name:** The name of the model file used for testing, e.g., "U-Net_20231218-1843.pkl".

**test_size:** A float representing the proportion of the dataset to be used for testing, e.g., 0.2.

**tile_size:** An integer defining the size of the tiles used in the experiment, e.g., 512.

## Model Settings

**train:** A dictionary containing training-specific parameters, such as:

**architecture:** The architecture of the model, e.g., "U-Net".

**backbone:** The backbone network used, e.g., "resnet18".

**epochs:** An integer indicating the number of training epochs, e.g., 30.

**loss_function:** Specifies the loss function used. It can be a string or None.

**training:** A boolean value (true or false) indicating whether the model is in the training phase.

## Usage

To use this configuration file:

**General Modification:** For broad changes applicable to all experiments, update the values directly in this template.

**Experiment-Specific Configuration:** For experiment-specific settings, make a copy of this template and modify the copied file accordingly.

**Running Experiments:** Use the configured YAML file with the experiment-running script or tool, ensuring that the script/tool is designed to read and apply these settings.

## Notes

Ensure that the paths and file names specified in the configuration file match those in your project directory.

Always validate the YAML file for syntax correctness after modification to avoid runtime errors. In case of issues, you can refer to an online YAML checker.
