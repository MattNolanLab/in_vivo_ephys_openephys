import open_ephys_IO
import os
import numpy as np

import PostSorting.open_field_make_plots


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
        first_opto_pulse_index = min(opto_on[0])
        prm.set_opto_tagging_start_index(first_opto_pulse_index)

    return opto_on, opto_off, is_found


# find first opto pulse and remove spatial data from that point
def remove_spatial_data_during_opto_stimulation(opto_on, spatial_data, prm):
    first_opto_pulse_index = prm.get_opto_tagging_start_index()
    sampling_rate_rate = prm.get_sampling_rate_rate()
    first_opto_pulse_index_bonsai = first_opto_pulse_index/sampling_rate_rate
    indices_to_remove = list(spatial_data.index[int(first_opto_pulse_index_bonsai):].values)
    spatial_data['without_opto_tagging'] = spatial_data.loc[first_opto_pulse_index:,''] = 16
    # PostSorting.open_field_make_plots.plot_position(spatial_data) # test
    return spatial_data
