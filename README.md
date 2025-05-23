# PurOja, recognising natural streams
Random Forest Classification, test to classify ditchlike streams. Balancing with SMOTE, stage: development project (deadline before finishing).

# First thing:
Open the top to bottom hierarchy document called: 01_basic_work_first_manuscript in 03_Manuscript/Documentation. It's the easiest way to follow the start of process and use these files in here. In the ReadMeFiles there is short descriptions or link to the source material needed. Python version used were Python 3.10 and 3.11. 

# Readme first, contents of the project explained
This directory is created based on a standard data analysis template (section 1. below, Jonas Hag). 
If you start from scratch:
1. To recreate the basic directory system before actual code or code files, please repeat: https://github.com/jonas-hag/analysistemplates
2. There has been some folders added and removed while working.  

## The file system of "PurOJa"-project
PurOja contents:
- README.md
- LICENSE.md
- 01_Data/01_raw
- 01_Data/01_raw/features (pickles, the feature material is saved in here)
- 01_Data/01_raw/features (npy files needed for working shared as external link)
- 01_Data/02_Clean
- 01_Data/02_Clean/data (resampled data based on the original study was in here)
- 01_Data/02_Clean/experiment_data (used in 01_terrain_indices_experiment is saved in here)
- 01_Data/02_Clean/03_Figures (images from 01_terrain indices images saved in here)
- 02_Analysis/01_scripts/ (arranged in numbered files and numbered scripts used)
- 02_Analysis/01_scripts/Functions (functions used in creating the database from the GEOTif-images)
- 02_Analysis/01_scripts/ 00_prework_files
- 02_Analysis/01_scripts/ 01_terrain_indices_experiment (one way to start: experimenting based on the Flyckt & al. documentation)
- 02_Analysis/01_scripts/ 02_view_functions (functions used in creating the experiment festures)
- 02_Analysis/02_Results
- 02_Analysis/03_Figures (RF visializations)
- 02_Analysis/04_Tables (empty)
- 03_Manuscript/01_basic_work_first_manuscript (Word, pdf, link to external location for downloading)
- 04_Presentation
- 05_Misc

## Supercomputer Puhti
Code for parallellized working in Puhti is at 02_Analysis/01_Scripts. The same code was used with 15 CPU's and 4 CPU's by adjusting the amount of cores. Code towards to CPU amount is in the zarr_processing.py file and in the final 02_Analysis/01_Scripts/03_RF_modelling file. 

## Working with Win 11
The file GDAL-3.4.3-cp311-cp311-win_amd64.whl was needed to work with GDAL (Geospatial Data Abstraction Library). Working with 4 CPU's and 64 Gb of RAM was also possible, but took more time. 

### AI
ChatGPT and Claude was used in reading the error messages and in solving the problems with coding. 

## Materials to read, or making searches about codes and techniques
- Whiteboxtools: https://www.whiteboxgeo.com/
- SAGA Gis Module Library Documentation: https://saga-gis.sourceforge.io/saga_tool_doc/2.3.0/index.html
- SAGA Gis is also in QGIS with Grass
- The Python Standard Library: https://docs.python.org/3/library/index.html

Original source article and contribution to the work done:

https://doi.org/10.1016/j.eswa.2022.116961 

https://figshare.com/projects/Detecting_ditches_using_machine_learning_on_high-resolution_DEMs/72779

Thanks to: JYU, CSC and Jonas Flyckt et al.