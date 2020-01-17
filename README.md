
Snakemake pipeline for processing open-ephys data

### Code structure
A workflow is defined in a snakefile. Currently, two snakefiles are provided, namely `vr_workflow.smk` and `op_workflow.smk` for VR and openfield experiment respectively.

Script files used in the workflow begin with a number (e.g. 01_sorting.py). They are numbered in the order of their execution.

Each script will read in some data file and always save some output in the end. The input and output files in each scripts are defined in the snakefile that uses them. All output are stored in a `processed` folder.

### How to use
The each script of the workflow can be run independently or as a workflow as a whole. Running each scirpt independently is great for debugging and development, while running the whole workflow is for batch processing.
- To open each script independently
    - Define the record to sort in the `debug_folder` in `setting.py` e.g. 
    
        ```
        debug_folder ='/home/ubuntu/to_sort/recordings/M5_2018-03-06_15-34-44_of
        ```

    - Just run `python <script name>` e.g. `python 01_sorting.py`

- To run it as a complete workflow
    - `snakemake --snakefile <snake file name> <recording folder>/processed/completed.txt` e.g.     
        ```
        snakemake --snakefile op_workflow.smk /home/ubuntu/to_sort/recordings/M5_2018-03-06_15-34-44_of/processed/completed.txt`
        ```


        or if you have put your files in `/home/ubuntu/to_sort/recordings`,
        simple run 
        
        ```
        snakemake --snakefile op_workflow.smk
        ```
        
        It will automatically run the pipeline in all recordings in that folder. It will also automatically skip the ones that are already processed.

### Common pitfalls
- **Snakemake doesn't rerun when you change the code** 

    When running the workflow as a whole, snakemake will determine automatically which part of the code needs to be rerun. Basically it checks the time of the output and input files, if the inputs are newers than the outputs, then it will run that script, otherwise it will skip it and continue with the next script.


### Tips and tricks
- If your recordings is too large to fit in memory, you can either use a larger instance or you can create a larger swapfile (recommanded) to be used as a temporary memory space:

    ```
    sudo fallocate -l 10G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    ```
- How to view plots on Eleanor

    If you are running the pipeline on a remote server, one the easiest way to 

### Known issues
- Always need to have all the files (e.g. dead_channels.txt), even if they are empty
- There should be a folder '/home/ubuntu/to_sort/recordings'


### Other problems
- Please file a issue for any problem that you have found!