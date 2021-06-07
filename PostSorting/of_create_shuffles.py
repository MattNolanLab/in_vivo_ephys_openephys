from pathlib import Path
import subprocess
import sys
import os

recording_folder_to_process = sys.argv[1]
#recording_folder_to_process = '/mnt/datastore/Harry/Cohort7_october2020/of'

local_scratch_path = '/exports/eddie/scratch/s1228823/recordings'
recordings_scratch_path = local_scratch_path+"/"+recording_folder_to_process.split("/")[-1]

ELEANOR_HOST = 'ubuntu@172.16.49.217'
ELEANOR_RECORDINGS_PATH = Path(recording_folder_to_process)
LOCAL_SCRATCH_PATH = Path(local_scratch_path)
N_SHUFFLES = 1000

# copy the recording file structure for getting the recording names
subprocess.check_call(f'rsync -avP --include "*/" --exclude="*" {ELEANOR_HOST}:{ELEANOR_RECORDINGS_PATH} {LOCAL_SCRATCH_PATH}', shell=True)

# get the recording names off the scratch space
recording_list = [f.path for f in os.scandir(recordings_scratch_path) if f.is_dir()]

for recording_name in recording_list:
    print(f'Copying spatial_firing.pkl and position.pkl from Datastore (via Eleanor) to Eddie Scratch')
    remote_recording_path = ELEANOR_RECORDINGS_PATH / recording_name
    local_recording_path = LOCAL_SCRATCH_PATH / recording_name
    subprocess.check_call(f"rsync -av {ELEANOR_HOST}:{remote_recording_path / 'MountainSort' / 'DataFrames' / 'spatial_firing.pkl'} {local_recording_path / 'MountainSort' / 'DataFrames' / 'spatial_firing.pkl'}", shell=True)
    subprocess.check_call(f"rsync -av {ELEANOR_HOST}:{remote_recording_path / 'MountainSort' / 'DataFrames' / 'position.pkl'} {local_recording_path / 'MountainSort' / 'DataFrames' / 'position.pkl'}", shell=True)

    print(f'Recordings copied... Submitting shuffle jobs to Eddie')
    for shuffle_number in range(N_SHUFFLES):
        print(f'Submitting shuffle job {shuffle_number}')
        cmd = f'qsub -pe sharedmem 2 -v RECORDING_PATH={local_recording_path} -v SHUFFLE_NUMBER={shuffle_number} /home/s1228823/in_vivo_ephys_openephys/PostSorting/shuffle_analysis.py'
        subprocess.check_call(cmd, shell=True)

print("all jobs submitted, good luck!")