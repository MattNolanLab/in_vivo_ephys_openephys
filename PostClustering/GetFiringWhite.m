function [spikeind,tetid,cluid,waveforms,whiteforms] = GetFiringWhite(separate_tetrodes,electrodes)
% reads the firings file and raw file and extracts:
% the information that would be in a .clu file
% the matrix of waveforms
% the matrix of waveforms from whitened data
% electrodes is an optional input in the form [2 3] if you want only
% tetrodes 2 and 3 for example

%% determine whether all_tetrodes, separate tetrodes, or both

if exist('separate_tetrodes','var')
    if separate_tetrodes==0
        disp('Running on all tetrodes')
        curpath{1}=('Electrophysiology/Spike_sorting/all_tetrodes/data/');
    elseif separate_tetrodes==1
        disp('Running on separate tetrodes')
        curpath{1}=('Electrophysiology/Spike_sorting/t1/data/');
        curpath{2}=('Electrophysiology/Spike_sorting/t2/data/');
        curpath{3}=('Electrophysiology/Spike_sorting/t3/data/');
        curpath{4}=('Electrophysiology/Spike_sorting/t4/data/');
    end
else
if exist('Electrophysiology/Spike_sorting/all_tetrodes/data/firings.mda','file')>0    
    disp('Running on all tetrodes')
    separate_tetrodes=0;
    curpath{1}=('Electrophysiology/Spike_sorting/all_tetrodes/data/');
elseif exist('Electrophysiology/Spike_sorting/t1/data/firings.mda','file')>0
    disp('Running on separate tetrodes')
    separate_tetrodes=1;
    curpath{1}=('Electrophysiology/Spike_sorting/t1/data/');
    curpath{2}=('Electrophysiology/Spike_sorting/t2/data/');
    curpath{3}=('Electrophysiology/Spike_sorting/t3/data/');
    curpath{4}=('Electrophysiology/Spike_sorting/t4/data/');
else
    disp('Error: No mda files found')
    return
end
end
if exist('electrodes','var') % only do this electrode
    if size(curpath)>1
        curpath=curpath(electrodes);
    end
else
    electrodes=1:length(curpath);
end
%% check that files exist
for p=1:length(curpath)  %1:length(curpath)
    fname=strcat(char(curpath(p)), 'firings.mda');
    if exist(fname,'file')==2
        getrid(p)=1;
    else
        electrodes(p)=[];
        getrid(p)=0;
    end
end
curpath=curpath(find(getrid==1));
%% open firings.mda
channel=[];spikeind=[];cluid=[];tetid=[];numclu=0;numprev=0;
for p=1:length(curpath)  %1:length(curpath)
fname=strcat(char(curpath(p)), 'firings.mda');
disp(strcat('Reading firings data - Tetrode= ',num2str(electrodes(p))));
firings=readmda(fname);
channeltet=firings(1,:);
spikeindtet=firings(2,:);
cluidtet=firings(3,:);
tetidtet=ones(size(cluidtet)).*electrodes(p);
clear firings

%% open raw.mda
fname=strcat(char(curpath(p)), 'filt.mda');
disp('Reading filtered waveform data');
raw=readmda(fname);
fname=strcat(char(curpath(p)), 'pre.mda');
disp('Reading whitened waveform data');
white=readmda(fname);

%% get rid of spikes right at the start of recording or right at the end (first 0.3msec + last msec)
channeltet(spikeindtet<10)=[]; channeltet(spikeindtet>length(raw)-30)=[];
cluidtet(spikeindtet<10)=[]; cluidtet(spikeindtet>length(raw)-30)=[];
tetidtet(spikeindtet<10)=[]; tetidtet(spikeindtet>length(raw)-30)=[];
spikeindtet(spikeindtet<10)=[]; spikeindtet(spikeindtet>length(raw)-30)=[];

if separate_tetrodes==1
    ch1=1; ch2=2; ch3=3; ch4=4;
for i=1:length(cluidtet)
    waveforms(1,1:40,numprev+i)=raw(ch1,spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(2,1:40,numprev+i)=raw(ch2,spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(3,1:40,numprev+i)=raw(ch3,spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(4,1:40,numprev+i)=raw(ch4,spikeindtet(i)-9:spikeindtet(i)+30);
    
    whiteforms(1,1:40,numprev+i)=white(ch1,spikeindtet(i)-9:spikeindtet(i)+30);
    whiteforms(2,1:40,numprev+i)=white(ch2,spikeindtet(i)-9:spikeindtet(i)+30);
    whiteforms(3,1:40,numprev+i)=white(ch3,spikeindtet(i)-9:spikeindtet(i)+30);
    whiteforms(4,1:40,numprev+i)=white(ch4,spikeindtet(i)-9:spikeindtet(i)+30);
end    
elseif separate_tetrodes==0
    tetidtet2=ceil(channeltet./4);
    ch1=1; ch2=2; ch3=3; ch4=4;
   %% need to think about this and most efficient way to structure multi-tetrode data
for i=1:length(cluidtet)
    
    waveforms(1,1:40,numprev+i)=raw(ch1+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(2,1:40,numprev+i)=raw(ch2+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(3,1:40,numprev+i)=raw(ch3+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(4,1:40,numprev+i)=raw(ch4+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    
    whiteforms(1,1:40,numprev+i)=white(ch1+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    whiteforms(2,1:40,numprev+i)=white(ch2+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    whiteforms(3,1:40,numprev+i)=white(ch3+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    whiteforms(4,1:40,numprev+i)=white(ch4+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
end        
end

channel=[channel channeltet];
spikeind=[spikeind spikeindtet];

if separate_tetrodes==0
tetid=[tetid tetidtet2];
else
tetid=[tetid tetidtet];
end

cluid=[cluid cluidtet+numclu];
numclu=max(cluid);
numprev=length(cluid);
end