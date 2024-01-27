# PSCAD and State Space Modelling Repository

This repository contains a collection of workspaces, scripts, and utilities related to PSCAD and State Space Modelling, particularly focusing on power system power flow and small-signal models. The structure is organized to facilitate easy access and modification of the scripts and models for the OEH (Operational Electric Hub).
## Repository Structure
### `pscad/`

This directory contains PSCAD workspaces, cases, and libraries. It is intended for storing all the PSCAD-related files and projects.
### `scripts/`

This directory is subdivided into several subdirectories, each serving a specific purpose in the realm of State Space Modelling.
#### `StateSpaceModelling/`

This is the main directory for all scripts related to State Space Modelling.
##### `PowerSystemPowerFlow/`

Contains scripts dedicated to calculating the power flow of the configured network of the OEH. These scripts are essential for analyzing the operational aspects of power systems.
##### `statespacemodels/`

This subdirectory contains various Excel files that represent different small-signal models. These models are crucial for understanding and simulating the behavior of power systems under small perturbations.
##### `script/`

Here, you'll find scripts written to assess the OEH and produce results. These scripts are integral to the analysis and interpretation of the system's performance and characteristics.

Additionally, this directory contains classes for the Component Connection Method (CCM) and other helper functions that are widely used in the scripts for modeling and analysis.

These scripts should be moved to `StateSpaceModels/` folder to work properly.

**NB: REMEMBER TO UPDATE PATHS** 
#### `utility/`

This directory houses a variety of scripts used for different plotting purposes. These utilities are helpful for visualizing data and results, making them crucial for effective presentation and interpretation of the analyses.
## Getting Started

To get started with this repository, clone it to your local machine and navigate to the respective directories for accessing the PSCAD workspaces, State Space Modelling scripts, or utility scripts as per your requirement.

Ensure you have the necessary environment and dependencies set up for running PSCAD simulations and Python scripts (if applicable).
