
Snakefile pipeline for processing open-ephys data

### Code structure
A workflow is defined in a snakefile. Currently, two snakefiles are provided, namely `vr_workflow.smk` and `op_workflow.smk` for VR and openfield experiment respectively.

Script files used in the workflow begin with a number (e.g. 01_sorting.py). They are numbered in the order of their execution.

Each script will read in some data file and always save some output in the end. The input and output files in each scripts are defined in snakefile that uses them.
