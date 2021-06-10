from pathlib import Path
import subprocess
import sys
import os

recording_folder_to_process = sys.argv[1] # something like /mnt/datastore/Harry/test_recordings or /mnt/datastore/Harry/Cohort7_october2020/of

local_scratch_path = '/exports/eddie/scratch/s1228823/recordings'
ELEANOR_HOST = 'ubuntu@172.16.49.217'
ELEANOR_RECORDINGS_PATH = Path(recording_folder_to_process)
LOCAL_SCRATCH_PATH = Path(local_scratch_path)
N_JOBS_PER_CELL = 1

# copy the recording file structure and the pickled dataframes
subprocess.check_call(f'rsync -avP --include "*/" --include="*.pkl" --exclude="*" {ELEANOR_HOST}:{ELEANOR_RECORDINGS_PATH} {LOCAL_SCRATCH_PATH}', shell=True)

# get the recording names off the scratch space
recordings_scratch_path = local_scratch_path+"/"+recording_folder_to_process.split("/")[-1]
recording_list = [f.path for f in os.scandir(recordings_scratch_path) if f.is_dir()]

n_jobs_submitted = 0
for recording_name in recording_list: # eg. M1_D1_2020-01-31_00-00-00
    print("looking for a shuffle.pkl here, ", recording_name+"/MountainSort/DataFrames/shuffles/shuffle.pkl")
    if os.path.isfile(recording_name+"/MountainSort/DataFrames/shuffles/shuffle.pkl"):
        print("shuffle.pkl was found for ", recording_name)
    else:
        print(f'Copying spatial_firing.pkl and position.pkl from Datastore (via Eleanor) to Eddie Scratch')
        remote_recording_path = ELEANOR_RECORDINGS_PATH / recording_name
        local_recording_path = LOCAL_SCRATCH_PATH / recording_name

        print(f'Recordings copied... Submitting shuffle jobs to Eddie')
        for job_i in range(N_JOBS_PER_CELL):
            print(f'Submitting shuffle job {job_i}')
            cmd = f'qsub -v RECORDING_PATH={local_recording_path} -v SHUFFLE_NUMBER={job_i} /home/s1228823/in_vivo_ephys_openephys/PostSorting/run_of_shuffle.sh'
            subprocess.check_call(cmd, shell=True)
            n_jobs_submitted += 1

print("A total of ", n_jobs_submitted, " have been submitted, good luck!")