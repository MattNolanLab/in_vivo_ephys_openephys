# in_vivo_ephys_openephys
![master](https://github.com/MattNolanLab/in_vivo_ephys_openephys/actions/workflows/test.yml/badge.svg)

Analysis for in vivo electrophysiology recordings saved in open ephys format. 

Snakemake pipeline for processing open-ephys data. This pipeline relies heavily on mountainsort4 and spikeinterface for sorting.


### Installation
An environment file is provided for all the dependencies. Please create the conda environment by 
```
conda env create -f environment.yml
```

After creating the new enviroment. Remember to do `conda init` and then `conda actviate ms4` to switch to the new environment. If set correctly, you should see the name of the environment in front of your command prompt.

Also plesae make sure you have gcc installed, if not please do `sudo apt-get install gcc` first. It is needed to compile the isosplit library for mountainsort4.

Due to the folder organization, you will also need to add the project root folder to your PYTHON path.
In the root folder of this repository i.e. `in_vivo_ephys_openephys`, execute the following. 

```export PYTHONPATH=$PYTHONPATH:$PWD``` 

You may want to add this into your `~/.bashrc` if you don't want to execute it everything you login your instance.

## Folder structure
A workflow is defined in a snakefile. Currently, two snakefiles are provided, namely `vr_workflow.smk` and `op_workflow.smk` for VR and openfield experiment respectively. The script files are in the `workflow` folder.

Each workflow defines a sequence of scripts to be called and the input dependences of each script. The scripts are located in the `scripts` folder. Inside there are sub-folder for each experiment type. Common scripts for all analysis (e.g. spike sorting) are in the top level. Each script begin with a number e.g. `01_xxx.py` that roughly indicate their order of execution. Some scripts can be run in parallel. The exact order of execution is determined by the dependencies graphs of scripts in the workflow files.

Each script will read in some data file and always save some output in the end. The input and output files in each scripts are defined in the workflow file that uses them. All outputs and figures are stored in the `processed` folder.

The pipeline assume you have a `parameters.yaml` in your recording folder, or in the parent folder where you recordings are located (to make it easy to define one parameter file for all recordings). It will try to search for the parameter file in the recording folder first, if not found, it will search for it in the parent folder.
At a minimum, the `parameters.yaml` file should contain the experimental type of the recording. Examples of parameter files can be found in `config/openfield` and `config/vr`. Each experiment type correponds to one snakemake workflow file. Their relation is defined in `config/workflow_config.yaml`. If you want to add a new experiment type, you just need to create a new workflow file and add the correspondance of the experiment type and snakemake workflow there.


## How to use


For typical use, the entry point of analysis is via the commandline interface defined in `runSnake.py`. You probably will need to change its permission to be executable first.

```
sudo chmod +x runSnake.py
```

You can get the list of argument to `runSnake` by 
```
./runSnake.py --help
```

For example, if you want to analyze a folder in datastore, you just need to run:

```
./runSnake.py /mnt/datastore/Someone/data/VR/M1_D1_2021-01
```
 By default, it will work on the datastore files directly. Some preliminary basic test has revealed no significant difference between working in the datastore directly or copying them to local drive first.

There is a `-n` option for dry-run, meaning that it will try simulate the workflow to see if there is any error without actually running it. It is always a good idea to try to use the `-n` option before you run the analysis on lots of data. i.e.

```
./runSnake.py -n /mnt/datastore/Someone/data/VR/M1_D1_2021-01
```
### Batch processing recordings

`runSnake` supports wild card in the folder name. Suppose your recordings are stored in `data/VR` and `data/openfield`, the following command will run the analysis on all experiment types, all animals, and all training day. The exact workflow to use for each recording will be determined by the `parameters.yaml` in the folder or its parent folder. **Remember** to use `-n` to double check first before doing analysis on lots of recording.

```
./runSnake.py /mnt/datastore/Someone/data/*/M?_D*
```
For structure of individual scripts and workflow, please consult the README in the `scripts` and `workflow` folders respectively.

### Only rerun a certain analysis
There are different analyzes defined as rule in the snakemake file. You can force it to rerun a certain analysis by using the `-R` option. If the analysis has already been completed fore, it will ask you to confirm if you really want to go forward. You can override this with the `--force` option

```
./runSnake.py /mnt/datastore/Junji/Data/2021cohort2/vr/M4_D44_2021-12-06_16-08-18 -R process_position --force
```

## Tips and tricks
- Snakemake insists on re-running the spike sorting
 
    By default, the pipeline will automatically skip analysis that has already been done (e.g. spike sorting) by checking the timestamp of the input and output files defined by a rule in the workflow file. But sometimes editing files in the folder may mess up the timestamp, always do a dry run with `-n` to double check. Sometimes, due to manual file editing, the pipeline may think that some of the input files are newer so it insist on re-runing them. If you want to avoid it, you can use the `--touch` option to touch all the output files and update their timestamps without actually running the analysis. Most time it will fix the issues.

    ```
    ./runSnake.py --touch /mnt/datastore/Someone/data/VR/M1_D1_2021-01
    ```
- Batch re-run of analysis

    The pipeline will skip recording that has alreay been processed. It will ask you if you want to re-run analysis on them, if you want to force re-run all analysis, you can use the `--force` flag. if you want to automatically skip processed recordings, you can use the `--skip` flag. If you have changed some rules and just want to rerun that particular and all subsequent rules that depends on it use the `-R` option with the name of the rule e.g. `-R process_position`


- If your recordings is too large to fit in memory, you can either use a larger instance or you can create a larger swapfile (recommanded) to be used as a temporary memory space:

    ```
    sudo fallocate -l 10G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    ```

### Other problems
- Please file a issue for any problem that you have found. Thank you!
