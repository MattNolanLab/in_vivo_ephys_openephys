
Snakemake pipeline for processing open-ephys data

### Package dependencies
- snakmake, spikeinterface, ml_ms4alg
- Any package that's required by the original pipeline

An environment file is provided for all the dependencies. Please create the conda environment by 
```
conda create -f environment.yml
```

note: make sure you have gcc installed, if not please do `sudo apt-get install gcc` first

### Code structure
A workflow is defined in a snakefile. Currently, two snakefiles are provided, namely `vr_workflow.smk` and `op_workflow.smk` for VR and openfield experiment respectively.

Script files used in the workflow begin with a number (e.g. 01_sorting.py). They are numbered in the order of their execution.

Each script will read in some data file and always save some output in the end. The input and output files in each scripts are defined in the snakefile that uses them. All outputs and plots are stored in the `processed` folder.

### How to use
The each script of the workflow can be run independently or as a workflow as a whole. Running each scirpt independently is great for debugging and development, while running the whole workflow is for batch processing.
- To run each script independently
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
        simply run 
        
        ```
        snakemake --snakefile op_workflow.smk
        ```
        
        It will automatically run the pipeline in all recordings in that folder. It will also automatically skip the ones that are already processed. The `processed/completed.txt` is just an empty file to signify to snakemake that it has completed all the workflow successfully.

### Common pitfalls
- **Snakemake doesn't rerun when you change the code** 

    When running the workflow as a whole, snakemake will determine automatically which part of the code needs to be rerun. Basically it checks the time of the output and input files, if the inputs are newer than the outputs, then it will run that script, otherwise it will skip it and continue with the next script. You can use the `--forceall` parameter when running snakemake to force it to rerun the analysis. 

    You won't have this problem when running each script independently.


### Tips and tricks
- If your recordings is too large to fit in memory, you can either use a larger instance or you can create a larger swapfile (recommanded) to be used as a temporary memory space:

    ```
    sudo fallocate -l 10G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    ```
- How to view plots on Eleanor

    The best way to view plot is to connect to the remove instance directly as a network drive.

    - Linux: Use [sshfs](https://askubuntu.com/questions/412477/mount-remote-directory-using-ssh)

    - Mac: [Mountain duck](https://mountainduck.io/) or [cyberduck](https://cyberduck.io/)

    - Windows: [SSH-FS Win Manager](https://github.com/evsar3/sshfs-win-manager)

- Executing cell blocks

    Each script contains sections marked by `%@@`. They indicate execution cells that can be run indepedently by supported editor like VS Code or [Pycharm via scientific mode](https://www.jetbrains.com/help/idea/matplotlib-tutorial.html#). These cell block allow you to debug and develop a specific function without rerunning the whole script. Try to utitlize it to keep yourself sane when developing new analysis.

### Known issues
- Always need to have all the files (e.g. dead_channels.txt), even if they are empty
- There should be a folder '/home/ubuntu/to_sort/recordings'
- Currently it doesn't read any experiment-specific parameters from the parameter file. It will be implemented later.

### Other problems
- Please file a issue for any problem that you have found. Thank you!
