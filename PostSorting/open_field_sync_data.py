import open_ephys_IO
import os
import numpy as np


def load_sync_data_ephys(recording_to_process, prm):
    is_found = False
    sync_data = None
    print('loading sync channel...')
    file_path = recording_to_process + '/' + prm.get_sync_channel()
    if os.path.exists(file_path):
        sync_data = open_ephys_IO.get_data_continuous(prm, file_path)
        is_found = True
    else:
        print('Opto data was not found.')
    return sync_data, is_found


def process_sync_data(recording_to_process, prm):
    sync_data, is_found = load_sync_data_ephys(recording_to_process, prm)
    return sync_data, is_found