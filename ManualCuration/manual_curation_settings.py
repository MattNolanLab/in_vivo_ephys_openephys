import os

"""
Manual curation instructions:
- Modify the path below. You do not need to modify anything else.
- Run manual_curation_pre_process to create the phy file for the recording you selected.
- Open the phy file on Eleanor and curate. Save the results in phy
- Run manual_curation_post_process to save your results on data store in a pipeline compatible format.
- Rerun the pipeline for this recording. It will see that there is curated data and skip the sorting and only run the
post-processing part.
"""

recording_name = 'M3_2021-05-26_14-19-02_of'
exp_folder = 'Klara/CA1_to_deep_MEC_in_vivo/analysis_test_manual/'
tetrode_geom_file = '/home/ubuntu/in_vivo_ephys_openephys/sorting_files/geom_all_tetrodes_original.csv'


def get_local_recording_path():
    main_local_folder = "/home/ubuntu/manual/"
    if not os.path.exists(main_local_folder):
        os.mkdir(main_local_folder)
    recording_local =main_local_folder + recording_name
    print('The manual curation files will be kept in: ' + recording_local)
    return recording_local


def get_recording_path_datastore():
    recording_server = "/mnt/datastore/" + exp_folder + recording_name
    return recording_server


def get_tetrode_geom_file_path():
    return tetrode_geom_file