import matplotlib.pylab as plt


def plot_position(position_data):
    plt.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=5)
    plt.show()


def plot_spikes_on_trajectory(position_data, spike_data):
    plt.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=10)
    cluster_id = 0
    plt.scatter(spike_data.position_x[0], spike_data.position_y[0], color='red', marker='o', s=30)
    plt.show()
    print('test')

