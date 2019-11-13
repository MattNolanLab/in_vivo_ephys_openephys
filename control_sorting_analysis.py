from joblib import Parallel, delayed
import gc
import glob
import os
import multiprocessing
import shutil
import subprocess
import sys
import traceback
import time
import Logger
from PreClustering import pre_process_ephys_data
from PostSorting import post_process_sorted_data
from PostSorting import post_process_sorted_data_vr

# set this to true if you want to skip the spike sorting step and use ths data from the server
skip_sorting = True

mountainsort_tmp_folder = '/tmp/mountainlab/'
sorting_folder = '/home/nolanlab/to_sort/recordings/'
to_sort_folder = '/home/nolanlab/to_sort/'
if os.environ.get('SERVER_PATH_FIRST_HALF'):
    server_path_first_half = os.environ['SERVER_PATH_FIRST_HALF']
    print(f'Using a custom server path: {server_path_first_half}')
else:
    # server_path_first_half = '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/'
    server_path_first_half = '/mnt/datastore/'
#server_path_first_half = 'smb://ardbeg.mvm.ed.ac.uk/nolanlab/'
#server_path_first_half = '/home/nolanlab/ardbeg/'
matlab_params_file_path = '/home/nolanlab/PycharmProjects/in_vivo_ephys_openephys/PostClustering/'
downtime_lists_path = '/home/nolanlab/to_sort/sort_downtime/'


def check_folder(sorting_path):
    recording_to_sort = False
    for dir, sub_dirs, files in os.walk(sorting_path):
        if not sub_dirs and not files:
            return recording_to_sort
        if not files:
            print('I am looking here: ', dir, sub_dirs)

        else:
            print('I found something, and I will try to sort it now.')
            recording_to_sort = find_sorting_directory()
            if recording_to_sort is False:
                return recording_to_sort
            else:
                return recording_to_sort


def find_sorting_directory():
    for name in glob.glob(sorting_folder + '*'):
        os.path.isdir(name)
        if check_if_recording_was_copied(name) is True:
            return name
        else:
            print('This recording was not copied completely, I cannot find copied.txt')
    return False


def check_if_recording_was_copied(recording_to_sort):
    if os.path.isfile(recording_to_sort + '/copied.txt') is True:
        return True
    else:
        return False


# return whether it is vr or openfield
def get_session_type(recording_directory):
    parameters_path = recording_directory + '/parameters.txt'
    try:
        param_file_reader = open(parameters_path, 'r')
        parameters = param_file_reader.readlines()
        parameters = list([x.strip() for x in parameters])
        session_type = parameters[0]

        if session_type == 'vr':
            is_vr = True
            is_open_field = False
        elif session_type == 'openfield':
            is_vr = False
            is_open_field = True
        else:
            print('Session type is not specified. '
                  'You need to write vr or openfield in the first line of the parameters.txt file. '
                  'You put {} there.'.format(session_type))
            is_vr = False
            is_open_field = False
    except Exception as ex:
        print('There is a problem with the parameter file.')
        print(ex)
    return is_vr, is_open_field


def get_location_on_server(recording_directory):
    parameters_path = recording_directory + '/parameters.txt'
    param_file_reader = open(parameters_path, 'r')
    parameters = param_file_reader.readlines()
    parameters = list([x.strip() for x in parameters])
    location_on_server = parameters[1]
    return location_on_server


def get_tags_parameter_file(recording_directory):
    tags = False
    parameters_path = recording_directory + '/parameters.txt'
    param_file_reader = open(parameters_path, 'r')
    parameters = param_file_reader.readlines()
    parameters = list([x.strip() for x in parameters])
    if len(parameters) > 2:
        tags = parameters[2]
    return tags


def write_param_file_for_matlab(file_to_sort, path_to_server, is_openfield, is_vr):
    if is_openfield:
        openfield = 1
    else:
        openfield = 0
    opto = 1
    params_for_matlab_file = open(matlab_params_file_path + "PostClusteringParams.txt", "w")
    params_for_matlab_file.write(file_to_sort + ',\n')
    params_for_matlab_file.write(server_path_first_half + path_to_server + ',\n')
    params_for_matlab_file.write(str(openfield) + ',\n')
    params_for_matlab_file.write(str(opto))
    params_for_matlab_file.close()


def write_shell_script_to_call_matlab(file_to_sort):
    script_path = file_to_sort + '/run_matlab.sh'
    batch_writer = open(script_path, 'w', newline='\n')
    batch_writer.write('#!/bin/bash\n')
    batch_writer.write('echo "-----------------------------------------------------------------------------------"\n')
    batch_writer.write('echo "This is a shell script that will call matlab."\n')
    batch_writer.write('export MATLABPATH=/home/nolanlab/PycharmProjects/in_vivo_ephys_openephys/PostClustering/\n')

    batch_writer.write('matlab -r PostClusteringAuto')


def check_if_matlab_was_successful(recording_to_sort):
    is_successful = True
    matlab_crash_path = recording_to_sort + '/matlabcrash.txt'

    if os.path.isfile(matlab_crash_path) is True:
        is_successful = False

    return is_successful


# write file 'crash_list.txt' in top level dir with list of recordings that could not be sorted
def add_to_list_of_failed_sortings(recording_to_sort):
    if os.path.isfile(to_sort_folder + "/crash_list.txt") is False:
        crash_writer = open(to_sort_folder + 'crash_list.txt', 'w', newline='\n')

    else:
        crash_writer = open(to_sort_folder + '/crash_list.txt', 'a', newline='\n')
    crashed_recording = str(recording_to_sort) + '\n'
    crash_writer.write(crashed_recording)
    crash_writer.close()


def call_matlab_post_sorting(recording_to_sort, location_on_server, is_open_field, is_vr):
    write_param_file_for_matlab(recording_to_sort, location_on_server, is_open_field, is_vr)
    write_shell_script_to_call_matlab(recording_to_sort)
    gc.collect()
    os.chmod(recording_to_sort + '/run_matlab.sh', 484)
    subprocess.call(recording_to_sort + '/run_matlab.sh', shell=True)

    if check_if_matlab_was_successful(recording_to_sort) is not True:
        raise Exception('Postprocessing failed, matlab crashed.')
    else:
        print('Post-processing in Matlab is done.')


def remove_folder_from_server_and_copy(recording_to_sort, location_on_server, name):
    if os.path.exists(server_path_first_half + location_on_server + name) is True:
        shutil.rmtree(server_path_first_half + location_on_server + name)
    try:
        if os.path.exists(recording_to_sort + name) is True:
            shutil.copytree(recording_to_sort + name, server_path_first_half + location_on_server + name)
    except shutil.Error as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        print('I am letting this exception pass, because shutil.copytree seems to have some permission issues '
              'I could not resolve, but the files are actually copied successfully.')
        pass


def copy_output_to_server(recording_to_sort, location_on_server):
    remove_folder_from_server_and_copy(recording_to_sort, location_on_server, '/Figures')
    remove_folder_from_server_and_copy(recording_to_sort, location_on_server, '/Firing_fields')
    remove_folder_from_server_and_copy(recording_to_sort, location_on_server, '/MountainSort')


def call_spike_sorting_analysis_scripts(recording_to_sort):

    try:
        is_vr, is_open_field = get_session_type(recording_to_sort)
        location_on_server = get_location_on_server(recording_to_sort)
        tags = get_tags_parameter_file(recording_to_sort)

        sys.stdout = Logger.Logger(server_path_first_half + location_on_server + '/sorting_log.txt')

        if not skip_sorting:
            pre_process_ephys_data.pre_process_data(recording_to_sort)

            print('I finished pre-processing the first recording. I will call MountainSort now.')
            os.chmod('/home/nolanlab/to_sort/run_sorting.sh', 484)

            subprocess.call('/home/nolanlab/to_sort/run_sorting.sh', shell=True)
            os.remove('/home/nolanlab/to_sort/run_sorting.sh')

            print('MS is done')

        # call python post-sorting scripts
        print('Post-sorting analysis (Python version) will run now.')
        if is_open_field:
            post_process_sorted_data.post_process_recording(recording_to_sort, 'openfield', running_parameter_tags=tags)
        if is_vr:
            post_process_sorted_data_vr.post_process_recording(recording_to_sort, 'vr', running_parameter_tags=tags)

        if os.path.exists(recording_to_sort + '/Figures') is True:
            copy_output_to_server(recording_to_sort, location_on_server)


        #call_matlab_post_sorting(recording_to_sort, location_on_server, is_open_field, is_vr)
        shutil.rmtree(recording_to_sort)
        if not skip_sorting:
            shutil.rmtree(mountainsort_tmp_folder)

    
    except Exception as ex:
        print('There is a problem with this file. '
              'I will move on to the next one. This is what Python says happened:')
        print(ex)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback)
        add_to_list_of_failed_sortings(recording_to_sort)
        location_on_server = get_location_on_server(recording_to_sort)
        if os.path.exists(recording_to_sort + '/Figures') is True:
            copy_output_to_server(recording_to_sort, location_on_server)

        shutil.rmtree(recording_to_sort)
        if os.path.exists(mountainsort_tmp_folder) is True:
            shutil.rmtree(mountainsort_tmp_folder)

        if os.environ.get('SINGLE_RUN'):
            print('Single run mode was active during the error. '
                  'I will quit immediately with a nonzero exit status instead of continuing to the next recording.')
            exit(1)  # an exit status of 1 means unsuccessful termination/program failure


def delete_processed_line(list_to_read_path):
    with open(list_to_read_path, 'r') as file_in:
        data = file_in.read().splitlines(True)
    with open(list_to_read_path, 'w') as file_out:
        file_out.writelines(data[1:])


def copy_file(filename, path_local):
    if os.path.isfile(filename) is True:
        if filename.split('.')[-1] == 'txt':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])
        if filename.split('.')[-1] == 'csv':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])
        if filename.split('.')[-1] == 'continuous':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])
        if filename.split('.')[-1] == 'pkl':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])
        if filename.split('.')[-1] == 'events':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])


def copy_recording_to_sort_to_local(recording_to_sort):
    path_server = server_path_first_half + recording_to_sort
    recording_to_sort_folder = recording_to_sort.split("/")[-1]
    path_local = sorting_folder + recording_to_sort_folder
    print('I will copy a folder from the server now. It will take a while.')
    if os.path.exists(path_server) is False:
        print('This folder does not exist on the server:')
        print(path_server)
        return False
    try:
        if os.path.exists(path_local) is False:
            os.makedirs(path_local)
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(copy_file)(filename, path_local) for filename in glob.glob(os.path.join(path_server, '*.*')))

        spatial_firing_path = path_server + '/MountainSort/DataFrames/spatial_firing.pkl'
        if os.path.isfile(spatial_firing_path) is True:
            if not os.path.isdir(path_local + '/MountainSort/DataFrames/'):
                os.makedirs(path_local + '/MountainSort/DataFrames/')
            shutil.copy(spatial_firing_path, path_local + '/MountainSort/DataFrames/spatial_firing.pkl')
        print('Copying is done, I will attempt to sort.')

    except Exception as ex:
        recording_to_sort = False
        add_to_list_of_failed_sortings(recording_to_sort)
        print('There is a problem with this file. '
              'I will move on to the next one. This is what Python says happened:')
        print(ex)
        return recording_to_sort
    return recording_to_sort


def get_next_recording_on_server_to_sort():
    recording_to_sort = False
    if not os.listdir(downtime_lists_path):
        return False
    else:
        list_to_read = os.listdir(downtime_lists_path)[0]
        list_to_read_path = downtime_lists_path + list_to_read
        if os.stat(list_to_read_path).st_size == 0:
            os.remove(list_to_read_path)
        else:
            downtime_file_reader = open(list_to_read_path, 'r+')
            recording_to_sort = downtime_file_reader.readlines()[0].strip()

            delete_processed_line(list_to_read_path)
            recording_to_sort = copy_recording_to_sort_to_local(recording_to_sort)
            if recording_to_sort is False:
                return False
            recording_to_sort = sorting_folder + recording_to_sort.split("/")[-1]

    return recording_to_sort


def monitor_to_sort():
    start_time = time.time()
    time_to_wait = 60.0
    while True:
        print('I am checking whether there is something to sort.')
        recording_to_sort = check_folder(sorting_folder)

        if recording_to_sort is not False:
            call_spike_sorting_analysis_scripts(recording_to_sort)

        else:
            if os.environ.get('SINGLE_RUN'):
                print('Single run mode was active, so I will exit instead of monitoring the folders.')
                break

            print('Nothing urgent to sort. I will check if there is anything waiting on the server.')

            recording_to_sort = get_next_recording_on_server_to_sort()
            if recording_to_sort is not False:
                call_spike_sorting_analysis_scripts(recording_to_sort)
            else:
                time.sleep(time_to_wait - ((time.time() - start_time) % time_to_wait))


def main():
    print('v - 0')
    print('-------------------------------------------------------------')
    print('This is a script that controls running the spike sorting analysis.')
    print('-------------------------------------------------------------')

    monitor_to_sort()


if __name__ == '__main__':
    main()