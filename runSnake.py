# Automatically run workflow in specify folder

import argparse
import sys
from snakemake import snakemake
from file_utility import parse_parameter_file
import os
import yaml
from pathlib import Path
from collections import defaultdict
import shutil
# import logging

# Command line argument
parser = argparse.ArgumentParser(description="Run snakemake workflow on specified folder")
parser.add_argument('Path', nargs='+', type=str, help='the recording to process')
parser.add_argument('--snakefile', '-f', help='specify the snakefile used to run the workflow. Otherwise it will determine the file to run automatically')
parser.add_argument('--watch','-w', action='store_true', default=False,help='Specify whether to monitor the folder for new recordings. Usually to be used with --all')
parser.add_argument('--all','-a', action='store_true', default=False, help='Whether to analyze all recordings under the folder')
parser.add_argument('--dryrun','-n', action= 'store_true', default=False, help='whether to dry-run the workflow')
parser.add_argument('--remotefiles','-r', action= 'store_true', default=False, help='Original files are in server. Will copy to local first')
parser.add_argument('--uploadresults','-u', action= 'store_true', default=False, help='whether to copy results to server')
parser.add_argument('--clean','-c', action= 'store_true', default=False, help='whether to delete local copy of recordings')
parser.add_argument('--overwrite','-o', action= 'store_true', default=False, help='whether to overwrite the processed files')


def _logPath(path,names):
    print(f'Working in {path}')
    return {}

args =  parser.parse_args()

# Load the workflow config file
config = yaml.load(open('config/workflow_config.yaml','r'), Loader=yaml.FullLoader)
local_recording_folder  = Path(config['local_recording_folder'])
result_folder = config['result_folder']

# Determine whether to search the whole folder
if args.all:
    paths = [p for p in Path(args.Path[0]).iterdir() if p.is_dir()]
else:
    paths = [Path(p) for p in  args.Path]

# Group the recordings according to their experiment type
targets = defaultdict(list)
for p in paths:
    param = yaml.load(open(p / 'parameters.yaml','r'), Loader=yaml.FullLoader)
    targets[param['expt_type']].append(p)



# Aggregate and execute files of of the same type together
for expt_type, paths in targets.items():
    print('==============================================')
    print(f'Running snakemake for file type: {expt_type}')
    print('==============================================')

    remote_folders = []

    #TODO: copying files should be work on folder and folder basis, to avoid downloading all the files at once
    # TODO: determine the best way to overwrite remote folder

    if args.remotefiles:
        for p in paths:
            # files is in remote server, need to copy to local folder first
            folderName = p.name
            remote_folders.append(p.parent)
            try:
                print('=======================================================')
                print(f'Copying {p} to {local_recording_folder / folderName}')
                if not args.dryrun:
                    shutil.copytree(p, local_recording_folder / folderName, ignore=_logPath)
                print('...complete')
                print('=======================================================')
            except FileExistsError:
                print('Folder already exists')


    #execute different workflow based on experiment type
    snakefile = config['snakefiles'][expt_type]

    # run the analysis on local copy
    if args.remotefiles:
        target_files = [str(local_recording_folder / p.name / 'processed' / 'snakemake.done')  for p in paths]
    else:
        target_files = [str(p/'processed/snakemake.done') for p in paths]

    # to do: enable multiple core processing
    snakemake(snakefile,targets = target_files,dryrun=args.dryrun, cores=config['cores'])

    # add check to verify the snakemake process is completed before uploading files
    # If need to upload results back to server
    if args.uploadresults:
        for i,p in enumerate(paths):
            local_path = Path(local_recording_folder) / p.name / result_folder
            print(f'Uploading {local_path} to {p}')
            if not args.dryrun:
                # pass
                try:
                    shutil.copytree(Path(local_recording_folder) / p.name / result_folder, p/result_folder)
                    print(f'...done')
                except FileExistsError:
                    print('...folder already exist, skipping...')
    
    # if need to clean up local folder
    if args.clean:
        for p in paths:
            local_path = Path(local_recording_folder) / p.name
            print(f'Removing {local_path}')
            if not args.dryrun:
                shutil.rmtree(local_path)
            print('done')

