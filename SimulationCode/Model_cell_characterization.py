import numpy as np
from matplotlib import pyplot as plt
import random
import math 
from neuron import h, gui
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import h5py
from os import listdir
from os.path import isfile
import csv
import glob
import os
import pandas as pd
import matplotlib.ticker as ticker
#import morphology of cell
h.xopen("./nolan/hocfile.hoc")

def generate_cell():
    single_cell=[]
    cell=h.Cell1()
    single_cell.append(cell)
return single_cell

def parameters(single_cell):
    h.celsius=37
    single_cell[0].soma.ena=45
    single_cell[0].soma.ek=-85

    single_cell[0].soma.gbar_kv = 10000 #potassium fast
    single_cell[0].soma.gbar_km = 15 #potassium slow
    single_cell[0].soma.gbar_na=72000 #sodium

    single_cell[0].soma.g_pas = 0.000033
    single_cell[0].soma.e_pas = -65 #v_init

    #all axon segments
    j=0
    for i in single_cell[0].axonal:
        single_cell[0].axon[j].gbar_kv =500
        single_cell[0].axon[j].gbar_km = 7
        single_cell[0].axon[j].gbar_na=1000
        single_cell[0].axon[j].ena=45
        single_cell[0].axon[j].ek=-85
        single_cell[0].axon[j].g_pas= 0.00001
        single_cell[0].axon[j].e_pas=-65
        j=j+1

    #all dendrite segments   
    j=0
    for i in single_cell[0].basal:
        single_cell[0].dend[j].g_pas=0.00001
        single_cell[0].dend[j].e_pas=-65
        j=j+1
return single_cell

def simulate (tstop):
    h.tstop=tstop
    h.run()
    
def set_recording_vectors(cell):
    soma_v_vec=h.Vector()
    t_vec=h.Vector()
    soma_v_vec.record(cell.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)
    return soma_v_vec, t_vec

def plot(single_cell):
    soma_v_vec, t_vec = set_recording_vectors(single_cell[0])
    simulate(tstop)
    soma_plot = plt.figure(figsize=(8,4))
    ax=soma_plot.add_subplot(111)
    ax.plot(t_vec, soma_v_vec, 'black', linewidth=1)
    plt.xlabel('time (ms)')
    plt.ylabel('voltage (mV)')
    loc = ticker.MultipleLocator(base=2)
    ax.yaxis.set_major_locator(loc)
    plt.show()
    return t_vec, soma_v_vec, soma_plot

def single_spike(single_cell):
    stimuli=[]
    stim = h.IClamp(single_cell[0].soma(0.5))
    stimuli.append(stim)
    stimuli[0].delay=20
    stimuli[0].dur=4
    stimuli[0].amp=1.5
    tstop=60
    
    t_vec, soma_v_vec, soma_plot=plot(single_cell)
    peak=max(soma_v_vec)
    res_t=t_vec[1]
    ind_baseline=int(20/res_t)
    baseline=soma_v_vec[ind_baseline] 
    half_max=(peak-baseline)/2+baseline
    i=soma_v_vec.indvwhere('>=', half_max)
    
    length=0
    for j in i:
        length=length+1

    half_width=t_vec[int(i[length-1])]-t_vec[int(i[0])]
    print('peak='+str(peak))
    print('half_width='+str(half_width)) 

def RMP(single_cell):    
    h.v_init=-65
    stimuli=[]
    stim = h.IClamp(single_cell[0].soma(0.5))
    stimuli.append(stim)
    stimuli[0].delay=100
    stimuli[0].dur=150
    stimuli[0].amp=0
    tstop=5000

    t_vec, soma_v_vec, soma_plot=plot(single_cell)
    t=np.asarray(t_vec.to_python())
    s=np.asarray(soma_v_vec.to_python())
    s[t>1000]
    RMP=np.mean(s)
    print('RMP='+str(RMP))
    
def voltage_thresh(single_cell):    
    h.v_init=-63
    stimuli=[]
    stim = h.IClamp(single_cell[0].soma(0.5))
    stimuli.append(stim)
    stimuli[0].delay=100
    stimuli[0].dur=150
    stimuli[0].amp=.083887512
    tstop=400

    t_vec, soma_v_vec, soma_plot=plot(single_cell)
    thresh=max(soma_v_vec)
    print('voltage_threshold='+str(thresh))
    
def sag_ratio(single_cell):
    h.v_init=-60
    stimuli=[]
    stim = h.IClamp(single_cell[0].soma(0.5))
    stimuli.append(stim)
    stimuli[0].delay=2000
    stimuli[0].dur=3000
    stimuli[0].amp=-0.040
    tstop=7000
    t_vec, soma_v_vec, soma_plot=plot(single_cell)
    soma=np.asarray(soma_v_vec.to_python())
    t=np.asarray(t_vec.to_python())
    
    soma_plot = plt.figure(figsize=(8,4))
    RMP=-63.12
    plt.plot(t[50000:-50000], soma[50000:-50000], color='k')
    plt.xlabel('time (ms)')
    plt.ylabel('voltage (mV)')
    mini=RMP-min(soma[50000:-50000])
    fr=int(t.shape[0]/7000*4000)
    to=int(t.shape[0]/7000*5000)
    base=soma[fr:to]
    meanb=RMP-np.mean(base)
    sag_r=meanb/mini
    print('sag_ratio='+str(sag_r))
    
def ramped_current_clamp(single_cell):
    stimuli=[]
    stim = h.IClamp(single_cell[0].soma(0.5))
    stimuli.append(stim)
    stim.delay=0
    stim.dur=1e9

    amps2=np.linspace(0,0.3,2000)
    amps2=np.asarray(amps2)
    amps1=np.zeros(int(1000))
    amps3=np.zeros(int(1000))
    amps=np.append([amps1],[amps2])
    amps=np.append([amps],[amps3])

    VecStim=h.Vector(amps) #eventvec
    input_plot=plt.figure(figsize=(8,2))
    ax=input_plot.add_subplot(111)
    plt.plot(amps*1000, 'black')
    plt.xlabel('time (ms)')
    plt.ylabel('current (pA)')
    loc = ticker.MultipleLocator(base=100) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.show()

    tstop=4000
    times=np.arange(0,5001,1)
    VecT=h.Vector(times)
    VecStim.play(stim._ref_amp, VecT)

    soma_v_vec, t_vec = set_recording_vectors(single_cell[0])
    simulate(tstop)
    soma_plot=plt.figure(figsize=(8,3))
    ax=soma_plot.add_subplot(111)
    ax.plot(t_vec, soma_v_vec, 'black')
    plt.xlabel('time (ms)')
    plt.ylabel('voltage (mV)')
    loc = ticker.MultipleLocator(base=40) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.show()

    high=int((soma_v_vec.indwhere('>=', 0)-int(0.5/res_t))*res_t)
    threshold=VecStim[high]
    print('rheobase='+str(threshold))

def depolarizing_block(single_cell):
    stimuli=[]
    stim = h.IClamp(single_cell[0].soma(0.5))
    stimuli.append(stim)
    stim.delay=0
    stim.dur=1e9

    amps2=np.linspace(0.015,100,3000)
    amps2=np.asarray(amps2)
    amps1=np.zeros(int(1000))
    amps3=np.zeros(int(1000))
    amps=np.append([amps1],[amps2])
    amps=np.append([amps],[amps3])

    VecStim=h.Vector(amps) #eventvec
    input_plot=plt.figure(figsize=(8,2))
    ax=input_plot.add_subplot(111)
    ax.plot(amps*1000, 'black')
    plt.xlabel('time (ms)')
    plt.ylabel('current (pA)')
    loc = ticker.MultipleLocator(base=30000) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.show()

    tstop=5000
    times=np.arange(0,5001,1)
    VecT=h.Vector(times)
    VecStim.play(stim._ref_amp, VecT)
    soma_v_vec, t_vec = set_recording_vectors(single_cell[0])
    simulate(tstop)
    soma_plot=plt.figure(figsize=(8,4))
    ax=soma_plot.add_subplot(111)
    ax.plot(t_vec, soma_v_vec, 'black', linewidth=1)
    plt.xlabel('time (ms)')
    plt.ylabel('voltage (mV)')
    loc = ticker.MultipleLocator(base=40) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.show()

    firing=soma_v_vec.c().indvwhere(soma_v_vec, '>=', 0) #where cell is firing
    depolarising_block=VecStim[int(firing[firing.size()-1]*res_t)]
    print('depolarising_block='+str(depolarising_block*1000))

    
def run_characterization():
    single_cell=generate_cell()
    single_cell=parameters(single_cell)    
    single_spike(single_cell) #for half-width and peak potential
    RMP(single_cell)
    voltage_thresh(single_cell)
    sag_ratio(single_cell)
    ramped_current_clamp(single_cell)
    depolarizing_block(single_cell)
   
