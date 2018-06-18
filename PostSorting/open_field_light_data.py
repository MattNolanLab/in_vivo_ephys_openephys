import open_ephys_IO
import os
import numpy as np


def load_opto_data(recording_to_process, prm):
    is_found = False
    opto_data = None
    print('loading opto channel...')
    file_path = recording_to_process + '/' + prm.get_opto_channel()
    if os.path.exists(file_path):
        opto_data = open_ephys_IO.get_data_continuous(prm, file_path)
        is_found = True
    else:
        print('Opto data was not found.')
    return opto_data, is_found


def get_ons_and_offs(opto_data):
    opto_on = np.where(opto_data > 0.5)
    opto_off = np.where(opto_data <= 0.5)
    return opto_on, opto_off


def process_opto_data(recording_to_process, prm):
    opto_on = opto_off = None
    opto_data, is_found = load_opto_data(recording_to_process, prm)
    if is_found:
        opto_on, opto_off = get_ons_and_offs(opto_data)

    return opto_on, opto_off, is_found

