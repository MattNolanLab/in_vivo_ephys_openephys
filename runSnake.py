# Automatically run workflow in specify folder

import argparse
import sys
from snakemake import snakemake
from file_utility import parse_parameter_file
import os
import yaml
from pathlib import Path
from collections import defaultdict

# Command line argument
parser = argparse.ArgumentParser(description="Run snakemake workflow on specified folder")
parser.add_argument('Path', nargs='+', type=str, help='the recording to process')
parser.add_argument('--snakefile', '-f', help='specify the snakefile used to run the workflow. Otherwise it will determine the file to run automatically')
parser.add_argument('--watch','-w', action='store_true', default=False,help='Specify whether to monitor the folder for new recordings. Usually to be used with --all')
parser.add_argument('--all','-a', action='store_true', default=False, help='Whether to analyze all recordings under the folder')
parser.add_argument('--dryrun','-n', action= 'store_true', default=False, help='whether to dry-run the workflow')
args =  parser.parse_args()

# Determine whether to search the whole folder
if args.all:
    paths = [p for p in Path(args.Path[0]).iterdir() if p.is_dir()]
else:
    paths = [Path(p) for p in  args.Path]

# Group the recordings according to their experiment type
targets = defaultdict(list)
for p in paths:
    param = parse_parameter_file(str(p/'parameters.txt'))# read parameter file
    targets[param['expt_type']].append(p)

# Load the workflow config file
config = yaml.load(open('config/workflow_config.yaml','r'), Loader=yaml.FullLoader)

# Aggregate and execute files of of the same type together
for expt_type, paths in targets.items():
    print('==============================================')
    print(f'Running snakemake for file type: {expt_type}')
    print('==============================================')

    targets = [str(p/'processed/snakemake.done') for p in paths]

    #execute different workflow based on experiment type
    snakefile = config['snakefiles'][expt_type]

    snakemake(snakefile,targets = targets,dryrun=args.dryrun)