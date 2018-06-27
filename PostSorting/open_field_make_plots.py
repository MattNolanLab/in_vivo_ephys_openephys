import matplotlib.pylab as plt


def plot_position(position_data):
    plt.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=5)
    plt.show()


def plot_spikes_on_trajectory(position_data, spike_data, cluster_id):
    plt.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=5)
