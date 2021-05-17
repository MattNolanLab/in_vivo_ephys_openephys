#!/usr/bin/env python

# Automatically run workflow in specify folder

import argparse
import sys
from snakemake import snakemake
from file_utility import parse_parameter_file, set_continuous_data_path
import os
import yaml
from pathlib import Path
from collections import defaultdict
import shutil
from glob import glob
import subprocess

# Add current directory to path so that the import function can work correct
subprocess.check_call('export PYTHONPATH=$PYTHONPATH:$PWD',shell=True)

# import logging

#TODO: add function to only rerun a particular rule

# Command line argument
parser = argparse.ArgumentParser(description="Run snakemake workflow on specified folder")
parser.add_argument('Path', nargs='+', type=str, help='the recording to process')
parser.add_argument('--snakefile', '-f', help='specify the snakefile used to run the workflow. Otherwise it will determine the file to run automatically')
parser.add_argument('--all','-a', action='store_true', default=False, help='Whether to analyze all recordings under the folder')
parser.add_argument('--dryrun','-n', action= 'store_true', default=False, help='whether to dry-run the workflow')
parser.add_argument('--remotefiles','-r', action= 'store_true', default=False, help='Original files are in server. Will copy to local first')
parser.add_argument('--uploadresults','-u', action= 'store_true', default=False, help='whether to copy results to server')
parser.add_argument('--clean','-c', action= 'store_true', default=False, help='whether to delete local copy of recordings')
parser.add_argument('--overwrite','-o', action= 'store_true', default=False, help='whether to overwrite the processed files')
parser.add_argument('--batch_process','-b', action= 'store_true', default=False, help='whether to download all recordings first and process them together.')
parser.add_argument('--force', action= 'store_true', default=False, help='whether to force rerun all analysis')
parser.add_argument('--skip','-s', action= 'store_true', default=False, help='whether skip processed recordings')
parser.add_argument('--unlock',action= 'store_true', default=False, help='whether to release the snakemake file lock')
parser.add_argument('--forcerun', '-R', nargs='*', help='whether to force run a certain rule')
parser.add_argument('--omit', '-O', nargs='*', help='whether to skip certain rule')
parser.add_argument('--touch', action= 'store_true', default=False, help='whether to touch files without running')


def _logPath(path,names):
    print(f'Working in {path}')
    return {}

args =  parser.parse_args()

# Load the workflow config file
config = yaml.load(open('config/workflow_config.yaml','r'), Loader=yaml.FullLoader)
local_recording_folder  = Path(config['local_recording_folder'])
result_folder = config['result_folder']

# Determine what recordings to process
if args.all:
    # process all subdirectories
    paths = [p for p in Path(args.Path[0]).iterdir() if p.is_dir()]
else:
    paths = []
    for p in args.Path:
        # use glob match if ? or * in path
        if '*' in p or '?' in p:
            paths += glob(p)
        else:
            paths.append(p)

    paths = [Path(p) for p in  paths] #convert all to Path


# Group the recordings according to their experiment type
targets = defaultdict(list)
for p in paths:
    param = None
    if (p / 'parameters.yaml').exists():
        param = yaml.load(open(p / 'parameters.yaml','r'), Loader=yaml.FullLoader)
    else:
        # try searching for the parent directory
        print('Parameter file not found in recording. Looking for it in the parent directory')
        if (p.parent / 'parameters.yaml').exists():
            param = yaml.load(open(p.parent / 'parameters.yaml','r'), Loader=yaml.FullLoader)
            print('Paramter file found in parent directory. I will use it now')
    
    if param is not None:
        targets[param['expt_type']].append(p)
    else:
        raise FileNotFoundError('Parameter file not found')



def download_files(paths, dryrun=True):
    # TODO: add feature to detect remote file has finished processing
    # download files from remote datastore
    remote_folders = []

    #TODO: copying files should be work on folder and folder basis, to avoid downloading all the files at once
    # TODO: determine the best way to overwrite remote folder

    for p in paths:
        # files is in remote server, need to copy to local folder first
        folderName = p.name
        remote_folders.append(p.parent)
        try:
            print(f'Copying {p} to {local_recording_folder / folderName}')
            if not dryrun:
                shutil.copytree(p, local_recording_folder / folderName, ignore=_logPath)
            print('...complete')
        except FileExistsError:
            print('Folder already exists')

    return remote_folders


def upload_files(paths, local_recording_folder,dryrun=True):
    # upload files to datastore
    # add check to verify the snakemake process is completed before uploading files
    # If need to upload results back to server

    for i,p in enumerate(paths):
        target_file = local_recording_folder / p.name / 'processed' / 'snakemake.done'
        if target_file.exists():
            local_path = Path(local_recording_folder) / p.name / result_folder
            print(f'Uploading {local_path} to {p}')
            if not dryrun:
                try:
                    shutil.copytree(Path(local_recording_folder) / p.name / result_folder, p/result_folder)
                    print(f'...done')
                except FileExistsError:
                    print('...folder already exist, skipping')
             
        else:
            print('Snakemake does not seem to be complete properly. Skipping upload')

def clean_up(paths, local_recording_folder, dryrun=True):
    # Delete the recording file after processing

    for p in paths:
        target_file = local_recording_folder / p.name / 'processed' / 'snakemake.done'
        local_path = Path(local_recording_folder) / p.name
        if target_file.exists():
            print(f'Removing {local_path}')
            if not args.dryrun:
                shutil.rmtree(local_path)
            print('done')
        else:
            if not args.dryrun:
                answer = input(f'Snakemake for {target_file} does not seem to complete successfully. Do you still want to remove it? (y/n) ')
                if answer == 'y':
                    print(f'Removing {local_path}')
                    if not dryrun:
                        shutil.rmtree(local_path)
                    print('done')
            else:
                print('In dryrun mode. Skipping upload')


def process_recordings(args, config, expt_type, paths):
    # processing the recording
    # include download, process, upload and clean up the recordings

    if args.remotefiles:
        remote_folders = download_files(paths,args.dryrun)

    #execute different workflow based on experiment type
    snakefile = config['snakefiles'][expt_type]

    # run the analysis on local copy
    if args.remotefiles:
        target_files = [str(local_recording_folder / p.name / 'processed' / 'snakemake.done')  for p in paths]
    else:
        target_files = [str(p/'processed/snakemake.done') for p in paths]

    # to do: enable multiple core processing
    snakemake(snakefile, targets = target_files, dryrun=args.dryrun, 
        cores=config['cores'], printreason=True, unlock=args.unlock, 
        forcerun=args.forcerun, omit_from=args.omit, touch=args.touch)


    if args.uploadresults:
        upload_files(paths,local_recording_folder,args.dryrun)

        # if needed to clean up local folder
        if args.clean and args.remotefiles:
            # safe-guarding: only delete local files when it is uploaded and the original recordings is in remote server
            # by default, it will delete files in the local_recording_folder only, it shouldn't delete
            # file in the remote server
            clean_up(paths, local_recording_folder,args.dryrun)

def filter_processed_recording(paths, isforce, will_upload, is_skip, isremote, dryrun=True):
    """
    Check if the recordig is already processed. Remove remote processed folder if necessary
    """
    # TODO: handle error when the remote has processed folder but no snakemake.done
    path2process = []
    for p in paths:
        if (p / 'processed' / 'snakemake.done').exists():

            if is_skip:
                print(f'I will skip {p} because it is already processed.')
                continue

            if not isforce:
                answer = input(f'{p} has already been processed. Do you want to process it again? (y/n) ')
                if answer =='y':
                    if will_upload:
                        answer2 = input('You have indicated you will upload the processed files. This will delete the remote processed folder. Are you sure? (y/n)')
                        if answer2 == 'y':
                            print('I will now remove remote processed folder')
                            if not dryrun:
                                shutil.rmtree(p/'processed')
                            remote_path = p/'processed'
                            print(f'...{remote_path} removed')
                            path2process.append(p)
                        else:
                            print('I will skip this recording.')
                    else:
                        path2process.append(p)
                else:
                    print('I will skip this recording.')
            else:
                # only remove file in forced mode when working with remote fle
                if (p/'processed').exists() and isremote:
                    if not dryrun:
                        shutil.rmtree(p/'processed')
                    
                    print(f'In forced mode, removed {remote_path}')
                    remote_path = p/'processed'
                path2process.append(p)
        else:
            path2process.append(p)

    return path2process

for expt_type, paths_all in targets.items():
    print('#######################################################################')
    print(f'############### Running snakemake for file type: {expt_type} #################')

    paths_all = filter_processed_recording(paths_all, args.force, args.uploadresults, args.skip, 
        args.remotefiles, args.dryrun)

    if args.batch_process:
        # download all files and process all at once (possibly faster but takes more space)
        paths = paths_all

        print('I will do batch processing for all of the following recordings simultaneously:')
        paths_list = '\n'.join([str(p) for p in paths])
        print(paths_list)
        process_recordings(args, config, expt_type, paths)
        print('############################################################')
    else:
        # do analysis recoridngs by recordings
        for path in paths_all:
            print(f'I will now process {path}')
            process_recordings(args, config, expt_type,[path])
            print('#######################################################')



