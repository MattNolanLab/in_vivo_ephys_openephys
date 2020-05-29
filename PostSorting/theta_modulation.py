import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pylab as plt
import math
import os
import pandas as pd
import PostSorting.parameters
import pickle
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import traceback
import warnings
import sys
warnings.filterwarnings('ignore')

test_params = PostSorting.parameters.Parameters()
import elephant as elephant

"""
https://elifesciences.org/articles/35949#s4
eLife 2018;7:e35949 DOI: 10.7554/eLife.35949
Kornienko et al., 2018
The theta rhythmicity of neurons was estimated from the instantaneous firing rate of the cell.
The number of spikes observed in 1 ms time window was calculated and convolved with a Gaussian kernel (standard deviation of 5 ms).
The firing probability was integrated over 2 ms windows and transformed into a firing rate.
A power spectrum of the instantaneous firing rate was calculated using the pwelchfunction of the oce R package.
The estimates of the spectral density were scaled by multiplying them by the corresponding frequencies: spec(x)∗freq(x).
A theta rhythmicity index for each neuron was defined as θ−baselineθ+baseline, where θ is the mean power at 6–10 Hz and baseline is the mean power in two adjacent frequency bands (3–5 and 11–13 Hz).
The theta rhythmicity indices of HD cells were analyzed with the Mclust function of the R package mclust which uses Gaussian mixture modeling and the EM algorithm to estimate the number of components in the data.
# obtain spike-time autocorrelations
windowSize<-300; binSize<-2
source(paste(indir,"get_stime_autocorrelation.R",sep="/"))
runOnSessionList(ep,sessionList=rss,fnct=get_stime_autocorrelation,
                 save=T,overwrite=T,parallel=T,cluster=cl,windowSize,binSize)
rm(get_stime_autocorrelation)
get_frequency_spectrum<-function(rs){
  print(rs@session)
  myList<-getRecSessionObjects(rs)
  st<-myList$st
  pt<-myList$pt
  cg<-myList$cg
  wf=c();wfid=c()
  m<-getIntervalsAtSpeed(pt,5,100)
  for (cc in 1:length(cg@id)) {
    st<-myList$st
    st<-setCellList(st,cc+1)
  ##########################################################
    st1<-setIntervals(st,s=m)
    st1<-ifr(st1,kernelSdMs = 5,windowSizeMs = 2)
    Fs=1000/2
    x=st1@ifr[1,]
    xts <- ts(x, frequency=Fs)
    w <- oce::pwelch(xts,nfft=512*2, plot=FALSE,log="no")
    wf0=w$spec*w$freq
    wf=rbind(wf,wf0)
    wfid=cbind(wfid,cg@id[cc])
  }
  return(list(spectrum=t(wf),spectrum.id=wfid,spectrum.freq=w$freq))
  }
  
##################################################################################
# calculate theta index from power spectra of instantaneous firing rates
freq=spectrum.freq[1,]
theta.i=c()
for (i in 1:dim(spectrum)[2]){
  wf=spectrum[,i]
  th=mean(wf[freq>6 & freq<10])
  b=mean(c(wf[freq>3 & freq<5],wf[freq>11 & freq<13]))
  ti=(th-b)/(th+b)
  theta.i=c(theta.i,ti)
}
x=theta.i[t$hd==1]
par(mfrow=c(2,3))
hist(x,main="Theta index distribution",ylab = "Number of cells", xlab="Theta index",15,xlim = c(-.05,.4),las=1)
x.gmm = Mclust(x)
x.s=summary(x.gmm)
print("Fit Gaussian finite mixture model")
print(paste("Number of components of best fit: ",x.s$G,sep=""))
print(paste("Log-likelhood: ",round(x.s$loglik,2),sep=""))
print(paste("BIC: ",round(x.s$bic,2),sep=""))
print("Theta index threshold = 0.07")
lines(c(0.07,0.07),c(0,14),col="red",lwd=2)
print(paste("Number of non-rhythmic (NR) HD cells (theta index threshold < 0.07): N = ",sum(x<.07),sep=""))
print(paste("Number of theta-rhythmic (TR) HD cells (theta index threshold > 0.07): N = ",sum(x>.07),sep=""))
##################################################################################
  
"""

def calculate_spectral_density(firing_rate, prm, spike_data, cluster, cluster_index, save_path):
    f, Pxx_den = signal.welch(firing_rate, fs=1000, nperseg=10000, scaling='spectrum')
    Pxx_den = Pxx_den*f
    #print(cluster)
    #plt.semilogy(f, Pxx_den)
    plt.plot(f, Pxx_den, color='black')
    plt.xlim(([0,20]))
    plt.ylim([0,max(Pxx_den)])
    #plt.ylim([1e1, 1e7])
    #plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_power_spectra_ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return f, Pxx_den


def calculate_firing_probability(convolved_spikes):
    firing_rate=[]
    firing_rate = get_rolling_sum(convolved_spikes, 2)
    return (firing_rate*1000)/2 # convert to Hz


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:] / window


def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window is too big, plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out

def extract_instantaneous_firing_rate(spike_data, cluster):
    firing_times=spike_data.at[cluster, "firing_times"]/30 # convert from samples to ms
    bins = np.arange(0,np.max(firing_times), 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    smoothened_instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)
    return smoothened_instantaneous_firing_rate

def calculate_theta_power(Pxx_den,f):
    theta_power = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f > 6, f < 10))))
    #baseline = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >  0, f < 50))))
    adjacent_power1 = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >= 3, f <=5))))
    adjacent_power2 = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >= 11, f <=13))))
    baseline = (adjacent_power1 + adjacent_power2)/2
    x = theta_power - baseline
    y = theta_power + baseline
    t_index = x/y
    return t_index, theta_power

def calculate_theta_index(spike_data,prm):
    print('I am calculating theta index...')
    save_path = prm.get_output_path() + '/Figures/firing_properties/autocorrelograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    theta_indices = []
    theta_powers = []
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1


        if len(spike_data.at[cluster, "firing_times"])<=1:
            # in the case no or 1 spike is found in open field or vr
            theta_indices.append(np.nan)
            theta_powers.append(np.nan)

        else:
            instantaneous_firing_rate = extract_instantaneous_firing_rate(spike_data, cluster)
            #convolved_spikes = convolve_spikes(instantaneous_firing_rate)
            firing_rate = calculate_firing_probability(instantaneous_firing_rate)
            f, Pxx_den = calculate_spectral_density(firing_rate, prm, spike_data, cluster, cluster_index, save_path)

            t_index, t_power = calculate_theta_power(Pxx_den, f)
            theta_indices.append(t_index)
            theta_powers.append(t_power)

            firing_times_cluster = spike_data.firing_times[cluster]
            corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 600)

            fig = plt.figure(figsize=(7,6)) # width, height?
            ax = fig.add_subplot(1, 2, 1)  # specify (nrows, ncols, axnum)
            ax.set_xlim(-300, 300)
            ax.plot(time, corr, '-', color='black')
            x=np.max(corr)
            ax.text(-200,x, "theta index = " + str(round(t_index,3)), fontsize =10)

            ax = fig.add_subplot(1, 2, 2)  # specify (nrows, ncols, axnum)
            ax.plot(f, Pxx_den, color='black')
            plt.ylim([0,max(Pxx_den)])
            plt.xlim(([0,20]))
            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('PSD [V**2/Hz]')
            x = max(Pxx_den)
            ax.text(3,x, "theta power = " + str(np.round(t_power,decimals=2)), fontsize =10)
            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_theta_properties.png', dpi=300)
            plt.close()

    spike_data["ThetaPower"] = theta_powers
    spike_data["ThetaIndex"] = theta_indices
    return spike_data




##################################################################################



def calculate_autocorrelogram_hist(spikes, bin_size, window):

    half_window = int(window/2)
    number_of_bins = int(math.ceil(spikes[-1]*1000))
    train = np.zeros(number_of_bins)
    bins = np.zeros(len(spikes))

    for spike in range(len(spikes)-1):
        bin = math.floor(spikes[spike]*1000)
        train[bin] = train[bin] + 1
        bins[spike] = bin

    counts = np.zeros(window+1)
    counted = 0
    for b in range(len(bins)):
        bin = int(bins[b])
        window_start = int(bin - half_window)
        window_end = int(bin + half_window + 1)
        if (window_start > 0) and (window_end < len(train)):
            counts = counts + train[window_start:window_end]
            counted = counted + sum(train[window_start:window_end]) - train[bin]

    counts[half_window] = 0
    if max(counts) == 0 and counted == 0:
        counted = 1

    corr = counts / counted
    time = np.arange(-half_window, half_window + 1, bin_size)
    return corr, time



def plot_autocorrelograms(spike_data, prm):
    plt.close()
    print('I will plot autocorrelograms for each cluster.')
    save_path = prm.get_output_path() + '/Figures/firing_properties/autocorrelograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        firing_times_cluster = spike_data.firing_times[cluster]
        #lags = plt.acorr(firing_times_cluster, maxlags=firing_times_cluster.size-1)
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 20)
        plt.xlim(-10, 10)
        plt.bar(time, corr, align='center', width=1, color='black')
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_autocorrelogram_10ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.figure()
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 500)
        plt.xlim(-250, 250)
        plt.bar(time, corr, align='center', width=1, color='black')
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_autocorrelogram_250ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()




def convolve_array(spikes):
    window = signal.gaussian(2, std=5)
    convolved_spikes = signal.convolve(spikes, window, mode='full')
    return convolved_spikes


#A theta rhythmicity index for each neuron was defined as θ−baselineθ+baseline,
## where θ is the mean power at 6–10 Hz and baseline is the mean power in two adjacent frequency bands (3–5 and 11–13 Hz).

"""
The theta index was calculated here as by Yartsev et al., (2011). 
First, we computed the autocorrelation of the spike train binned by 0.01 seconds with lags up to ±0.5 seconds. 
Without normalization, this may be interpreted as the counts of spikes that occurred in each 0.01 second bin 
after a previous spike (Figure 1a). The mean was then subtracted, and the spectrum was calculated as the square 
of the magnitude of the fast-Fourier transform of this signal, zero-padded to 216 samples. 
This spectrum was then smoothed with a 2-Hz rectangular window (Figure 1b), 
and the theta index was calculated as the ratio of the mean of the spectrum within 1-Hz of each side of the 
peak in the 5-11 Hz range to the mean power between 0 and 50 Hz.
"""

def run_test(spatial_firing, id=None):
    if id is not None:
        spatial_firing = spatial_firing[spatial_firing["cluster_id"] == id]

    if "ThetaIndex" in list(spatial_firing):
        spatial_firing = spatial_firing.drop(columns=["ThetaIndex"])
    if "ThetaPower" in list(spatial_firing):
        spatial_firing = spatial_firing.drop(columns=["ThetaPower"])

    spatial_firing = calculate_theta_index(spatial_firing, test_params)
    return spatial_firing

def gen_random_firing(n_clusters):
    print("done")
    spatial_firing = pd.DataFrame()
    cluster_ids = []
    firing_times = []
    session_ids = []
    for i in range(n_clusters):
        cluster_ids.append(i+1)
        firing_times.append(np.sort(np.random.choice(100000000, 10000, replace=False)))
        session_ids.append("random_firing_test")
    spatial_firing["firing_times"] = firing_times
    spatial_firing["cluster_id"] = cluster_ids
    spatial_firing["session_id"] = session_ids
    return spatial_firing

def gen_modulated_firing(n_clusters, freq=8):
    print("done")

    total_length = 1000000
    n_spikes = 10000
    F = freq # desired frequnecy
    Fs = 30000 # sampling rate
    T = total_length/Fs # n seconds   30000
    Ts = 1./Fs
    N = int(T/Ts)
    t = np.linspace(0, T, N)
    signal = np.cos(2*np.pi*F*t)+1
    signal_normalised = signal/np.sum(signal)

    spatial_firing = pd.DataFrame()
    cluster_ids = []
    firing_times = []
    session_ids = []
    for i in range(n_clusters):
        cluster_ids.append(i+1)
        firing_times.append(np.sort(np.random.choice(total_length, n_spikes, replace=False, p=signal_normalised)))
        session_ids.append("random_firing_test")
    spatial_firing["firing_times"] = firing_times
    spatial_firing["cluster_id"] = cluster_ids
    spatial_firing["session_id"] = session_ids
    return spatial_firing

def run_for_x(path_to_recording_list):
    recordings_file_reader = open(path_to_recording_list, 'r')
    recordings = recordings_file_reader.readlines()
    list_of_recordings = list([x.strip() for x in recordings])

    for i in range(len(list_of_recordings)):
        try:
            recording_path = list_of_recordings[i]
            spatial_firing_path = recording_path + "/MountainSort/DataFrames/spatial_firing.pkl"
            spatial_firing = pd.read_pickle('/mnt/datastore/'+spatial_firing_path)
            spatial_firing= spatial_firing.sort_values(by=['cluster_id'])
            test_params.set_output_path('/mnt/datastore/'+recording_path+"/MountainSort")
            test_params.set_sampling_rate(30000)
            spatial_firing = run_test(spatial_firing)
            spatial_firing.to_pickle('/mnt/datastore/'+spatial_firing_path)

        except Exception as ex:
            print("failed on recording, ", list_of_recordings[i])
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)

def main():

    '''
    recording_path = '/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField/M2_D10_2019-06-28_14-56-06'
    spatial_firing_path = recording_path + "/MountainSort/DataFrames/spatial_firing.pkl"
    spatial_firing = pd.read_pickle(spatial_firing_path)
    spatial_firing= spatial_firing.sort_values(by=['cluster_id'])
    test_params.set_output_path(recording_path+"/MountainSort")
    test_params.set_sampling_rate(30000)
    spatial_firing = run_test(spatial_firing)
    spatial_firing.to_pickle(spatial_firing_path)
    '''

    #run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField/of_list.txt")
    #run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/vr_list.txt")

    run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/vrlist.txt")
    run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild/mouse_info/filelist_m2.txt")
    run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild/mouse_info/filelist_m3.txt")

    run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/vrlist_cohort3.txt")
    run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild/Mouse_Info/filelist_M1_of.txt")
    run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild/Mouse_Info/filelist_M6_of_1.txt")

    run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/with_of_recordings.txt")
    run_for_x("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/OpenFeild/Mouse_info/filelist_all.txt")

    # generate a dummy spatial firing dataframe to test if theta index can come out low if made randomly.
    #random_firing = gen_modulated_firing(n_clusters=3, freq=8)
    #random_firing = gen_random_firing(n_clusters=3)
    #random_firing = run_test(random_firing)
    print("look now")

if __name__ == '__main__':
    main()