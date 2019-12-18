function [spikeind,tetid,cluid,waveforms] = GetFiring(separate_tetrodes,SortingComputer,electrodes)
% reads the firings file and raw file and extracts:
% the information that would be in a .clu file
% the matrix of waveforms
% electrodes is an optional input in the form [2 3] if you want only
% tetrodes 2 and 3 for example
%% get dead channels
if exist('dead_channels.txt','file')>0
fid=fopen('dead_channels.txt');
dead_channels=fscanf(fid,'%d');
fclose(fid);
else
    dead_channels=[];
end
%% determine whether all_tetrodes, separate tetrodes, or both
if exist('separate_tetrodes','var')
    if separate_tetrodes==0
        disp('Running on all tetrodes')
        if SortingComputer==1
            curpath{1}=('Electrophysiology/Spike_sorting/all_tetrodes/data/');
        else
            curpath{1}=('mdafiles/all_');
        end
    elseif separate_tetrodes==1
        disp('Running on separate tetrodes')
        if SortingComputer==1
            curpath{1}=('Electrophysiology/Spike_sorting/t1/data/');
            curpath{2}=('Electrophysiology/Spike_sorting/t2/data/');
            curpath{3}=('Electrophysiology/Spike_sorting/t3/data/');
            curpath{4}=('Electrophysiology/Spike_sorting/t4/data/');
        else
            curpath{1}=('mdafiles/T1_');
            curpath{2}=('mdafiles/T2_');
            curpath{3}=('mdafiles/T3_');
            curpath{4}=('mdafiles/T4_');
        end
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
    elseif exist('mdafiles/all_firings.mda','file')>0
        disp('Running on all tetrodes')
        separate_tetrodes=0;
        curpath{1}=('mdafiles/all_');
    elseif exist('mdafiles/T1.mda','file')>0
        disp('Running on separate tetrodes')
        separate_tetrodes=1;
        curpath{1}=('mdafiles/T1_');
        curpath{2}=('mdafiles/T2_');
        curpath{3}=('mdafiles/T3_');
        curpath{4}=('mdafiles/T4_');
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

%% open firings.mda
channel=[];spikeind=[];cluid=[];tetid=[];numclu=0;numprev=0;
for p=1:length(electrodes)  %1:length(curpath)
fname=strcat(char(curpath(electrodes(p))), 'firings.mda');
disp(strcat('Reading firings data - Tetrode= ',num2str(electrodes(p))));
firings=readmda(fname);
channeltet=firings(1,:);
spikeindtet=firings(2,:);
cluidtet=firings(3,:);
tetidtet=ones(size(cluidtet)).*electrodes(p);
clear firings
%% open raw.mda
fname=strcat(char(curpath(electrodes(p))), 'filt.mda');
disp('Reading filtered waveform data');
raw=readmda(fname);
%% correct for dead channels
if size(dead_channels)>0
    dead_channels_tet=[];
    if separate_tetrodes==1 %do this for separate tetrodes
        if any(ceil(dead_channels/4)==electrodes(p))% check if there is a dead channel on this tetrode
        dead_channels_tet=dead_channels(ceil(dead_channels/4)==electrodes(p));
        dead_channels_tet=dead_channels_tet-((electrodes(p)-1)*4);
        lives=1:4;
        lives(dead_channels_tet)=[]; % fix raw 
        raw2=NaN(size(raw,1)+length(dead_channels),size(raw,2));
        raw2(lives,:)=raw;
        raw=raw2; % fix raw to compensate for dead channels
        clear raw2;
        for add=1:length(dead_channels_tet) %fix channeltet to compensate for dead channels
            dead=dead_channels_tet(add);
            channeltet(channeltet>=dead)=channeltet(channeltet>=dead)+1;
        end
        end
    elseif separate_tetrodes==0
        lives=1:16;
        lives(dead_channels)=[]; % fix raw 
        raw2=NaN(size(raw,1)+length(dead_channels),size(raw,2));
        raw2(lives,:)=raw;
        raw=raw2; % fix raw to compensate for dead channels
        clear raw2;
        for add=1:length(dead_channels) %fix channeltet to compensate for dead channels
            dead=dead_channels(add);
            channeltet(channeltet>=dead)=channeltet(channeltet>=dead)+1;
        end
    end    
end
% fname=strcat(char(curpath(p)), 'pre.mda');
% disp('Reading whitened waveform data');
% white=readmda(fname);

%% get rid of spikes right at the start of recording or right at the end (first 0.3msec + last msec)
channeltet(spikeindtet<10)=[]; channeltet(spikeindtet>length(raw)-30)=[];
cluidtet(spikeindtet<10)=[]; cluidtet(spikeindtet>length(raw)-30)=[];
tetidtet(spikeindtet<10)=[]; tetidtet(spikeindtet>length(raw)-30)=[];
spikeindtet(spikeindtet<10)=[]; spikeindtet(spikeindtet>length(raw)-30)=[];

if separate_tetrodes==1
    for i=1:length(cluidtet)
    waveforms(1,1:40,numprev+i)=raw(1,spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(2,1:40,numprev+i)=raw(2,spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(3,1:40,numprev+i)=raw(3,spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(4,1:40,numprev+i)=raw(4,spikeindtet(i)-9:spikeindtet(i)+30);
    
%     whiteforms(1,1:40,numprev+i)=white(ch1,spikeindtet(i)-9:spikeindtet(i)+30);
%     whiteforms(2,1:40,numprev+i)=white(ch2,spikeindtet(i)-9:spikeindtet(i)+30);
%     whiteforms(3,1:40,numprev+i)=white(ch3,spikeindtet(i)-9:spikeindtet(i)+30);
%     whiteforms(4,1:40,numprev+i)=white(ch4,spikeindtet(i)-9:spikeindtet(i)+30);
end    
elseif separate_tetrodes==0
    tetidtet2=ceil(channeltet./4);
    
   %% need to think about this and most efficient way to structure multi-tetrode data
for i=1:length(cluidtet)
    
    waveforms(1,1:40,numprev+i)=raw(1+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(2,1:40,numprev+i)=raw(2+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(3,1:40,numprev+i)=raw(3+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    waveforms(4,1:40,numprev+i)=raw(4+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
    
%     whiteforms(1,1:40,numprev+i)=white(ch1+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
%     whiteforms(2,1:40,numprev+i)=white(ch2+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
%     whiteforms(3,1:40,numprev+i)=white(ch3+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
%     whiteforms(4,1:40,numprev+i)=white(ch4+((tetidtet2(i)-1)*4),spikeindtet(i)-9:spikeindtet(i)+30);
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