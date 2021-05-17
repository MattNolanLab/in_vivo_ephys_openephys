import os
import matplotlib.pylab as plt
from utils import power_spectra


def plot_power_spectrum_for_hd(freqs, idx, ps, figure_path):
    # save_path = prm.get_local_recording_folder_path() + prm.get_sorter_name() + '/Figures/hd_power_spectrum'
    # if os.path.exists(save_path) is False:
    #     os.makedirs(save_path)
    plt.xlim(0, 15)
    plt.xlabel('Frequencies [Hz]')
    plt.plot(freqs[idx], ps[idx])
    # '/' + 'hd_power_spectrum.png'
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def check_if_hd_sampling_was_high_enough(spatial_data, figure_path):
    freqs, idx, ps = power_spectra.power_spectrum(spatial_data.hd)
    plot_power_spectrum_for_hd(freqs, idx, ps,figure_path)
