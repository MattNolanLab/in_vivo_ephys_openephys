import open_ephys_IO
import os
import matplotlib.pylab as plt


def load_opto_data(recording_to_process, prm):
    is_found = False
    opto_data = None
    print('loading opto channel...')
    file_path = recording_to_process + '/' + prm.get_opto_channel()
    if os.path.exists(file_path):
        opto_data = open_ephys_IO.get_data_continuous(prm, file_path)
        plt.plot(opto_data)
        plt.show()
        is_found = True
    else:
        print('Opto data was not found.')
    return opto_data, is_found


def process_opto_data(recording_to_process, prm):
    opto_data, is_found = load_opto_data(recording_to_process, prm)