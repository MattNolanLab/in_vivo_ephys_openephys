function PostClusteringAuto(SortingComputer)
% POSTCLUSTERINGAUTO produces summary figures for each cluster produced by
% MountainSort. It can open bonsai or axona position files and sync them.
%
% if SortingComputer is set to 1 it requires a file in the matlab folder
% called PostClusteringParams.txt This has 4 lines of contents.
% 1. path is path to folder on sorting computer
% 2. outfile is path to folder on server
% 3. OpenField is either 1 if 2d pos data is present, or 0 if not
% 4. Opto is either 1 if there are light pulses, or 0 if not
% If matlab crashes it will return control to the shell script on the
% sorting computer. This can be disabled by commenting out the
% try-catch-loop for debugging purposes.
% it should auto-detect problems with missing mda files, opto files and pos
% files and skip affected plots%
try
    if ~exist('SortingComputer','var')
        SortingComputer=1; % 1 if running on SortingComputer, else 0
    end
    copy=0; % set to 0 unless you want to copy the mda files to the datastore
    %% find input parameters
    if SortingComputer==1
        [errormessage, outfile, OpenField, Opto] = find_input_parameters();
        %% copy mda files to server - slow and large files so not recommended unless needed
        if copy==1
            errormessage = copy_mda_files_to_server(outfile);
        end
    else
        outfile=pwd;
        OpenField=1;
        Opto=1;
    end

    %% Autodetect GSQ and Axona - these variables specify what the position data and pixel ratio will be
    'png'; % what format to save the output figures (matlab figure saved anyway)
    electrodes=0; % set as 0 if using all tetrodes

    %% Set subplot locations for Figure 2
    [fig_rows, fig_cols, waveplotstile, postile, ratemaptile, gridcortile, posmaptile, hdpolartile, postilerun, ratemaptilerun, gridcortilerun, posmaptilerun, optoplotstile, optotile, optohisttile, refractoryperiodtile, rasterplottile, thetacorrtile, speedtile] = set_subplots_for_figure();
    %% get session name from the directory name
    [errormessage, sessionid] = get_session_name(SortingComputer);
    %% Get position data
    [errormessage,OpenField,post,posx,posy,hd,HDint] = get_position_data(OpenField, Axona, GSQ, pixel_ratio);
    %% Get Real-TimeStamps + light pulse data
    [errormessage, LEDons, LEDoffs, timestamps, Opto] = get_time_stamps_and_light_pulses(Opto);
    stage={'all tetrodes', 'individual tetrodes'}; % variable for informative error message
    %% Repeat rest of script for both all-tetrodes and separate tetrodes
    for separate_tetrodes=0:1 % make plots for both all-tetrodes and separate
        if ~exist(strcat('Figures',num2str(separate_tetrodes)),'dir')
            mkdir(strcat('Figures',num2str(separate_tetrodes))); % create folder for figures
        end
        %% Get Spike Data
        [errormessage, spikeind, tetid, cluid, waveforms, numclu] = get_spike_data(separate_tetrodes, electrodes, SortingComputer);
        %% trim spiking data to the length of the original data
        [errormessage, tetid, cluid, waveforms, spikeind] = trim_spiking_data_to_length_of_original_data(tetid, spikeind, timestamps, cluid, waveforms, separate_tetrodes);
        %% trim pos data to length of ephys data
        [posx, posy, post, hd, HDint] = trim_position_data_to_length_of_ephys(OpenField, posx, posy, post, hd, HDint,timestamps);
        %% Get real-time timestamps for each spike
        [total_time, spiketimes] = get_real_time_timestamps_for_spikes(timestamps, spikeind);
        %% Trim position data to remove it during light stimulation
        [posx, posy, hd, HDint, post] = remove_position_data_during_light_stimulation(OpenField, Opto, posx, posy, post, LEDons, hd, HDint);
        %% Calculate Running speed
        [errormessage, speed, posxrun, posyrun, postboundary, sampling_rate] = get_running_speed(OpenField, separate_tetrodes,posx,posy,post,pixel_ratio,speed_cut);
        %% Make Output figures
        errormessage=strcat('making figures - ',char(stage(separate_tetrodes+1)));
        for i=1:numclu
            [datamatrix] = process_cell(separate_tetrodes, i, spiketimes, cluid, tetid, waveforms,OpenField,postboundary,post,posx,posy,hd,posxrun,posyrun,speed, fig_rows, fig_cols, waveplotstile,total_time,rasterplottile,refractoryperiodtile,thetacorrtile,postile,pixel_ratio,ratemaptilerun,posmaptilerun,ratemaptile,posmaptile,postilerun,speed_cut,gridcortile,gridcortilerun,hdpolartile,speedtile,sampling_rate,Opto,LEDons,LEDoffs,optotile,GSQ,optohisttile,optoplotstile,sessionid,format);       
            datasave(i,:)=datamatrix;
        end
        errormessage=strcat('saving data matrix - ',char(stage(separate_tetrodes+1)));
        if separate_tetrodes==0
            save('datasave0','datasave'); % save data matrix with info about each cluster
        elseif separate_tetrodes==1
            save('datasave1','datasave'); % save data matrix with info about each cluster
        end
        %copy snippet data to server
        [errormessage] = copy_snippet_data_to_server(SortingComputer,separate_tetrodes,outfile);
    end
    %% copy data to server
    [errormessage] = copy_data_to_server(SortingComputer, outfile);
    if SortingComputer==1; clear variables; disp('finished running matlab script, returning control to python'); exit; end % exit so automatic script can continue on next dataset
catch
    disp(strcat('Matlab script failed_',errormessage));
    fid = fopen( 'matlabcrash.txt', 'a' );
    fprintf(fid,strcat('Matlab crashed while_',(errormessage)));
    fclose(fid);
    try
        in='matlabcrash.txt';
        out=strcat(outfile,'/matlabcrash.txt');
        copyfile(in,out);
    catch
    end
    if SortingComputer==1
        [errormessage] = try_to_copy_data();
    clear variables;
    disp('returning control to python');
    exit;
    end
end % exit so automatic script can continue on next dataset
%end

function[errormessage, outfile, OpenField, Opto] = find_input_parameters()
    errormessage='inital variables'; % This variable is needed for outputting an error message to Pycharm
    fid=fopen('home/nolanlab/PycharmProjects/in_vivo_ephys_openephys/PostClustering/PostClusteringParams.txt','r');
    params=fscanf(fid,'%s');
    params=split(params,',');
    outfile=strcat(char(params(2)),',',char(params(3))); % outfile is the path to coresponding datastore folder
    path=params(1); path=char(path); % path is the path to the data on the sorting computer
    OpenField=double(params(4)); % should be 1 if there is position data, else 0
    Opto=double(params(5)); % should be 1 if there are 3msec opto-tagging pulses, else 0
    cd(path); % change directory to data folder
end

function[errormessage] = copy_mda_files_to_server(outfile)
    errormessage='copying files to server';
    disp('Copying mda files to datastore');
    mkdir(outfile,'mdafiles');
    innames={'all_tetrodes','t1','t2','t3','t4'};
    outnames={'all','T1','T2','T3','T4'};
    for i=1:length(innames) % loop to copy the filt.mda files
        in=strcat('Electrophysiology/Spike_sorting/',char(innames(i)),'/data/filt.mda');
        out=strcat(outfile,'/mdafiles/',char(outnames(i)),'_filt.mda');
        if exist(in,'file')
            copyfile(in,out);
        end
    end
    for i=1:length(innames) % loop to copy the firings.mda files
        in=strcat('Electrophysiology/Spike_sorting/',char(innames(i)),'/data/firings.mda');
        out=strcat(outfile,'/mdafiles/',char(outnames(i)),'_firings.mda');
        if exist(in,'file')
            copyfile(in,out);
        end
    end
end

function[GSQ, Axona, pixel_ratio] = get_acq_system_and_pixel_ratio()
%% Autodetect GSQ and Axona - these variables specify what the position data and pixel ratio will be
date=pwd;
date=date(strfind(pwd,'_201')+1:strfind(pwd,'_201')+10);
serialdate=datenum(date,'yyyy-mm-dd');
if serialdate<736847 % GSq axona tracking
    GSQ=1; Axona=1; pixel_ratio=490;
elseif serialdate>736847 && serialdate<737030 % GSq bonsai
    GSQ=1; Axona=0; pixel_ratio=490;
else % HRB bonsai tracking
    GSQ=0; Axona=0; pixel_ratio=440;
end
end

function[fig_rows,fig_cols,waveplotstile,postile,ratemaptile,gridcortile,posmaptile,hdpolartile,postilerun,ratemaptilerun,gridcortilerun,posmaptilerun,optoplotstile,optotile,optohisttile,refractoryperiodtile,rasterplottile,thetacorrtile,speedtile] = set_subplots_for_figure()
    %% Subplot locations for Figure 2
    % these set the positions for each subplot in the final figures
    fig_rows=6; fig_cols=10;    % 6 by 10 grid for subplots
    waveplotstile=[1 2 11 12];      % waveform plots
    postile=[3 4 13 14];            % position plot
    ratemaptile=[5 6 15 16];        % rate map
    gridcortile=[7 8 17 18];        % grid plot
    posmaptile=[9 10 19 20];        % coverage plot
    hdpolartile=[21 22 31 32];      % head direction
    postilerun=[23 24 33 34];       % running position plot
    ratemaptilerun=[25 26 35 36];   % running rate map
    gridcortilerun=[27 28 37 38];   % running grid plot
    posmaptilerun=[29 30 39 40];    % running coverage plot
    optoplotstile=[41 42 51 52];    % opto-tagged waveforms
    optotile=[43 44 53 54];         % opto-raster
    optohisttile=[45 46 55 56];     % opto- histogram
    refractoryperiodtile=[47 48];   % spike-time autocorrelogram zoom
    rasterplottile=[49 50];         % firing rate v time
    thetacorrtile=[57 58];          % spike-time autocorrelogram theta
    speedtile = [59 60];            % firing rate v running speed
end

function[errormessage,sessionid] = get_session_name(SortingComputer)
    %% get session name from the directory name
    errormessage='finding session name';
    str = pwd; %full path to this directory
    if SortingComputer==1
        idx = strfind(str,'/') ; % i changed this from \
    else
        idx=strfind(str,'\');
    end
    foldername = str(idx(end)+1:end);
    idx=strfind(foldername,'_');
    try
        animal=foldername(1:idx(1)-1); % if foldername starts with an animal id, find it
        date=foldername(idx(1)+1:idx(2)-1); % find the date from the foldername
    catch
        animal='unknown'; %else if no animal id set it to unknown
        date=foldername(1:idx(1)-1); % find the date from the folder name
    end
    sessionid=strcat(animal,'-', num2str(date));
end

function[errormessage,OpenField,post,posx,posy,hd,HDint] = get_position_data(OpenField,Axona,GSQ, pixel_ratio)
   %% Get position data
   date=pwd;
   date=date(strfind(pwd,'_201')+1:strfind(pwd,'_201')+10);
    errormessage='getting position data';
    if OpenField==1 % only do this if it's an open field session
        try
            if Axona==0
                posnameTiz=strcat(num2str(date(6:7)),num2str(date(9:10)));
                posnameTiz=dir(strcat('*',posnameTiz,'*')); posnameTiz=posnameTiz.name;
                if any(size(dir('Tracking*'),1))%% if it's Klara's tracking file
                    pname=dir('Tracking*'); pname=pname.name;
                    [post,posx,posy,hd,HDint]= GetPosSyncedCorr(pname,pixel_ratio);
                elseif any(size(dir(posnameTiz),1))
                    if GSQ==1%% bonsai George Square
                        pname=posnameTiz;
                        [post,posx,posy,hd,HDint]= GetPosSynced(pname,pixel_ratio);
                    else
                        pname=posnameTiz;
                        [post,posx,posy,hd,HDint]= GetPosSyncedCorr(pname,pixel_ratio);
                    end
                else
                    disp('Error: no bonsai position data file');
                    OpenField=0;
                end
            elseif any(size(dir('*.POS'),1)) %% if it's the axona position tracking system
                [post,posx,posy,hd,HDint]= GetPosSyncedAxona;
            else %% to pick up errors
                disp('Error: no axona position data file');
                OpenField=0;
            end
        catch
            disp('Error: problem finding position data file. Trying without position');
            OpenField=0;
        end
    end
end

function[errormessage, LEDons, LEDoffs, timestamps, Opto] = get_time_stamps_and_light_pulses(Opto)
    %% Get Real-TimeStamps + light pulse data
    errormessage='getting optotagging data';
    if Opto==1 % only do this if it's an opto-tagging session

        try
            [LEDons,LEDoffs,timestamps]=GetOpto;
            crash=LEDons(1); % makes it move to the catch if size LEDons<0 - no pulse data
        catch
            disp('Problem with Opto data. Skipping these plots')
            try
                [~, timestamps, ~] = load_open_ephys_data('105_CH1_0.continuous');
            catch
                [~, timestamps, ~] = load_open_ephys_data('100_CH1.continuous');
            end
            Opto=0;
        end
    else
        try
            [~, timestamps, ~] = load_open_ephys_data('105_CH1_0.continuous');
        catch
            [~, timestamps, ~] = load_open_ephys_data('100_CH1.continuous');
        end
    end
end

function[errormessage, spikeind, tetid, cluid, waveforms, numclu] = get_spike_data(separate_tetrodes, electrodes, SortingComputer)
    %% Get Spike Data
    errormessage=strcat('getting spike data - ',char(separate_tetrodes+1));
    try
       load(strcat('Firings',num2str(separate_tetrodes),'.mat'));
       size(spikeind);
       size(tetid);
       size(cluid);
       size(waveforms);
    catch
       if electrodes==0
            [spikeind,tetid,cluid,waveforms] = GetFiring(separate_tetrodes,SortingComputer); %open mda files and create Firings.mat file
            save(strcat('Firings',num2str(separate_tetrodes),'.mat'),'empty','spikeind','tetid','cluid','waveforms','-v7.3');
        else % run on specific tetrodes only if specified in initial variables
            [spikeind,tetid,cluid,waveforms] = GetFiring(separate_tetrodes,SortingComputer,electrodes);
            save(strcat('Firings',num2str(separate_tetrodes),'.mat'),'empty','spikeind','tetid','cluid','waveforms','-v7.3');
       end
    end
    numclu=length(unique(cluid)); % how many clusters are present
    numtet=length(unique(tetid)); % how many tetrodes have been analysed
    disp(strcat('Found-',num2str(numclu),' clusters, across-',num2str(numtet),' tetrodes' ))
end

function[errormessage, tetid, cluid, waveforms, spikeind] = trim_spiking_data_to_length_of_original_data(tetid, spikeind, timestamps, cluid, waveforms,separate_tetrodes)
    %% trim spiking data to the length of the original data
    % for some reason the python .continuous reader has an extra 1024 samples
    % than the matlab .continuous reader so we need to remove these extra
    % samples if the mda file was created using the python openephys reader
    tetid(spikeind>length(timestamps))=[];
    cluid(spikeind>length(timestamps))=[];
    waveforms(:,:,spikeind>length(timestamps))=[];
    %whiteforms(:,:,spikeind>length(timestamps))=[];
    spikeind(spikeind>length(timestamps))=[];
    errormessage=strcat('combining spike and position data - ',char(separate_tetrodes+1));
end

function[posx, posy, post, hd, HDint] = trim_position_data_to_length_of_ephys(OpenField, posx, posy, post, hd, HDint, timestamps)
%% trim position data to length of ephys data
        if OpenField==1
            posx=posx(post>min(timestamps) & post<max(timestamps));
            posy=posy(post>min(timestamps) & post<max(timestamps));
            hd=hd(post>min(timestamps) & post<max(timestamps));
            HDint=HDint(post>min(timestamps) & post<max(timestamps));
            post=post(post>min(timestamps) & post<max(timestamps));
        end
end

function[total_time, spiketimes] = get_real_time_timestamps_for_spikes(timestamps, spikeind)
    %% Get real-time timestamps for each spike
    total_time=max(timestamps)-min(timestamps);
    spiketimes=timestamps(spikeind); % realtimestamps for each spike
end

function[posx, posy, hd, HDint, post] = remove_position_data_during_light_stimulation(OpenField, Opto, posx, posy, post, LEDons, hd, HDint)
    %% Trim position data to remove it during light stimulation
    if OpenField==1 && Opto==1
        posx=posx(post<(min(LEDons)-60));
        posy=posy(post<(min(LEDons)-60));
        hd=hd(post<(min(LEDons)-60));
        HDint=HDint(post<(min(LEDons)-60));
        post=post(post<(min(LEDons)-60));
    end
end

function[errormessage, speed, posxrun, posyrun, postboundary, sampling_rate] = get_running_speed(OpenField,separate_tetrodes,posx,posy,post,pixel_ratio, speed_cut)
    if OpenField==1 % only do this if it's an open field session
        errormessage=strcat('calculating running speed - ',char(separate_tetrodes+1));
        %% make running position data only
        % filter by running speed
        [runind,speed]=speedfilter(posx,posy,post,speed_cut,pixel_ratio); % speed in cm/sec
        posxrun=posx; posxrun(runind==0)=NaN;
        posyrun=posy; posyrun(runind==0)=NaN;
        %% set bins for working out closest position sample
        postboundary=post-[max(diff(post)); diff(post)./2];
        postboundary=[postboundary; max(postboundary)+(max(diff(post))/2)];
        sampling_rate=1/mean(diff(post)); %average position sampling rate. Needed for converting firing rates to Hz
    end
end



function[errormessage] = copy_snippet_data_to_server(SortingComputer,separate_tetrodes,outfile)
    errormessage=strcat('copying snippets to server');
    if SortingComputer==1 % copy snippet data to server
        if exist(strcat('Firings',num2str(separate_tetrodes),'.mat'),'file')>0
            disp('copying snippet data to server');
            copyfile(strcat('Firings',num2str(separate_tetrodes),'.mat'),strcat(outfile,'/Firings',num2str(separate_tetrodes),'.mat'));
        end
    end
end

function[errormessage] = copy_data_to_server(SortingComputer, outfile)
    %% copy data to server
    errormessage=strcat('copying figures to server');
    if SortingComputer==1
        %% copy figures and data matrix to server
        disp('copying figures to server');
        copyfile('datasave0.mat',strcat(outfile,'/datasave_all.mat'));
        copyfile('datasave1.mat',strcat(outfile,'/datasave_separate.mat'));
        copyfile('Figures0/*.fig',strcat(outfile,'/SortingFigures_all_M'));
        copyfile('Figures0/*.png',strcat(outfile,'/SortingFigures_all_PNG'));
        copyfile('Figures1/*.fig',strcat(outfile,'/SortingFigures_separate_M'));
        copyfile('Figures1/*.png',strcat(outfile,'/SortingFigures_separate_PNG'));
        %% copy cluster metrics to server
        disp('Copying metrics files to datastore');
        mkdir(outfile,'clustermetrics');
        innames={'all_tetrodes','t1','t2','t3','t4'};
        outnames={'all','T1','T2','T3','T4'};
        for i=1:length(innames)
            in=strcat('Electrophysiology/Spike_sorting/',char(innames(i)),'/cluster_metrics.json');
            out=strcat(outfile,'/clustermetrics/',char(outnames(i)),'_cluster_metrics.json');
            if exist(in,'file')
                copyfile(in,out);
            end
        end
    end
end

function[errormessage] = try_to_copy_data(outfile)
    separate_tetrodes = 0;
    errormessage=strcat('copying figures to server');
    try
        if exist(strcat('Firings',num2str(separate_tetrodes),'.mat'),'file')>0
            disp('copying snippet data to server');
            for separate_tetrodes=0:1
                copyfile(strcat('Firings',num2str(separate_tetrodes),'.mat'),strcat(outfile,'/Firings',num2str(separate_tetrodes),'.mat'));
            end
        else
            disp('Copying mda files to datastore');
            mkdir(outfile,'mdafiles');
            innames={'all_tetrodes','t1','t2','t3','t4'};
            outnames={'all','T1','T2','T3','T4'};
            for i=1:length(innames) % loop to copy the filt.mda files
                in=strcat('Electrophysiology/Spike_sorting/',char(innames(i)),'/data/filt.mda');
                out=strcat(outfile,'/mdafiles/',char(outnames(i)),'_filt.mda');
                if exist(in,'file')
                    copyfile(in,out);
                end
            end
            for i=1:length(innames) % loop to copy the firings.mda files
                in=strcat('Electrophysiology/Spike_sorting/',char(innames(i)),'/data/firings.mda');
                out=strcat(outfile,'/mdafiles/',char(outnames(i)),'_firings.mda');
                if exist(in,'file')
                    copyfile(in,out);
                end
            end
        end
        disp('Copying metrics files to datastore');
        mkdir(outfile,'clustermetrics');
        innames={'all_tetrodes','t1','t2','t3','t4'};
        outnames={'all','T1','T2','T3','T4'};
        for i=1:length(innames)
            in=strcat('Electrophysiology/Spike_sorting/',char(innames(i)),'/cluster_metrics.json');
            out=strcat(outfile,'/clustermetrics/',char(outnames(i)),'_cluster_metrics.json');
            if exist(in,'file')
                copyfile(in,out);
            end
        end
    catch
        disp('copying backup files to server failed');
    end
end

function[datamatrix] = process_cell(separate_tetrodes, i, spiketimes, cluid, tetid, waveforms,OpenField,postboundary,post,posx,posy,hd,posxrun,posyrun,speed, fig_rows, fig_cols, waveplotstile,total_time,rasterplottile,refractoryperiodtile,thetacorrtile,postile,pixel_ratio,ratemaptilerun,posmaptilerun,ratemaptile,posmaptile,postilerun,speed_cut,gridcortile,gridcortilerun,hdpolartile,speedtile,sampling_rate,Opto,LEDons,LEDoffs,optotile,GSQ,optohisttile,optoplotstile,sessionid,format)
    figure2=figure;
    set(gcf,'color','white');
    %% get data for this cluster only
    [errormessage, cluspktimes, nspikes, tet, cluwaves] = get_data_for_cluster(separate_tetrodes, i, spiketimes, cluid, tetid, waveforms);

    %% find spike positions - this will only run for open field sessions
    [errormessage,spkx,spky,spkhd,spkxrun,spkyrun,spkspeed] = find_spike_positions(OpenField,cluspktimes, postboundary, post,i,separate_tetrodes,posx,posy,hd,posxrun,posyrun,speed);

    %% make all the subplots
    %% spike plots - waveforms spike-time autocorrelograms and firing rate against time
    [errormessage, max_ampwaves, max_channelwaves, spk_widthwaves, ori] = make_spike_plots(separate_tetrodes, i, cluwaves, fig_rows, fig_cols, waveplotstile, cluspktimes, total_time, rasterplottile, refractoryperiodtile, thetacorrtile);
    %% position plots - only for open field
    [errormessage,skaggs,spars,cohe,max_firing,coverage,skaggsrun,sparsrun,coherun,max_firingrun,grid_score,grid_scorerun,frh_hd,meandir_hd,r_hd] = make_position_plots(OpenField,separate_tetrodes,i,posx,posy,spkx,spky,fig_rows,fig_cols,postile,pixel_ratio,post,ratemaptilerun,posmaptilerun,ratemaptile,posmaptile,posxrun,posyrun,spkxrun,spkyrun,postilerun,speed_cut,gridcortile,gridcortilerun,hdpolartile,speedtile,hd,spkhd,sampling_rate,cluspktimes,speed,spkspeed);
    %% optoplots - waveforms, raster, and firing rate histogram
    [errormessage,lightscore_p,lightscore_I,lightlatency,percentresponse,lightscore_p2,lightscore_I2,lightlatency2,percentresponse2,lightscore_p3,lightscore_I3,lightlatency3,percentresponse3,lightscore_p4,lightscore_I4,lightlatency4,percentresponse4] = make_opto_plots(Opto,separate_tetrodes,i,cluspktimes,LEDons,LEDoffs,fig_rows,fig_cols,optotile,GSQ,cluwaves,optohisttile,optoplotstile,ori);

    %% save plot and store data matrix
    [errormessage] = save_plots(separate_tetrodes,i,figure2,sessionid,tet,format);
    %save data
    if Opto==0; lightscore_p=NaN; lightscore_I=NaN; lightlatency=NaN; percentresponse=NaN; lightscore_p2=NaN; lightscore_I2=NaN; lightlatency2=NaN; percentresponse2=NaN; lightscore_p3=NaN; lightscore_I3=NaN; lightlatency3=NaN; percentresponse3=NaN; lightscore_p4=NaN; lightscore_I4=NaN; lightlatency4=NaN; percentresponse4=NaN; end % fill output data
    if OpenField==0; frh_hd=NaN; meandir_hd=NaN; r_hd=NaN; skaggs=NaN; spars=NaN; cohe=NaN; max_firing=NaN; grid_score=NaN; skaggsrun=NaN; sparsrun=NaN; coherun=NaN; max_firingrun=NaN; grid_scorerun=NaN; coverage=NaN; end; %fill output data

    errormessage=strcat('making data matrix - ',char(separate_tetrodes+1),'cluster - ',num2str(i));
    datamatrix=[tet,i,nspikes,coverage,nspikes/total_time, max_ampwaves, max_channelwaves,spk_widthwaves, max(frh_hd), meandir_hd, r_hd, skaggs,spars,cohe,max_firing,grid_score,skaggsrun,sparsrun,coherun,max_firingrun,grid_scorerun,lightscore_p,lightscore_I,lightlatency,percentresponse,lightscore_p2,lightscore_I2,lightlatency2,percentresponse2,lightscore_p3,lightscore_I3,lightlatency3,percentresponse3,lightscore_p4,lightscore_I4,lightlatency4,percentresponse4];
    
end

function[errormessage, cluspktimes, nspikes, tet, cluwaves] = get_data_for_cluster(separate_tetrodes, i, spiketimes, cluid, tetid, waveforms)
    errormessage=strcat('collecting cluster-specific spikes - ',char(separate_tetrodes+1),'cluster - ',num2str(i));
    cluspktimes=spiketimes(cluid==i); nspikes=length(cluspktimes); % calculate number of spikes
    tet=tetid(cluid==i);
    if max(tet)==min(tet); tet=max(tet); else; tet=mode(tet); disp('Error, cluster not on one tetrode');end
    cluwaves=waveforms(:,:,cluid==i);
    %cluwhites=whiteforms(:,:,cluid==i);

end

function[errormessage,spkx,spky,spkhd,spkxrun,spkyrun,spkspeed] = find_spike_positions(OpenField,cluspktimes, postboundary, post,i,separate_tetrodes,posx,posy,hd,posxrun,posyrun,speed)
    if OpenField==1 % only do this if it's an open field session
        %% find spike positions
        errormessage=strcat('getting spike pos data - ',char(separate_tetrodes+1),'cluster - ',num2str(i));
        [~,posspk]=histc(cluspktimes,postboundary);
        posspk(posspk==0)=length(post);
        spkx=posx(posspk);
        spky=posy(posspk);
        spkhd=hd(posspk);
        spkxrun=posxrun(posspk);
        spkyrun=posyrun(posspk);
        spkspeed=speed(posspk);
        spkx(cluspktimes>max(post))=NaN; spkx(cluspktimes<min(post))=NaN;
        spky(cluspktimes>max(post))=NaN; spky(cluspktimes<min(post))=NaN;
        spkhd(cluspktimes>max(post))=NaN; spkhd(cluspktimes<min(post))=NaN;
        spkxrun(cluspktimes>max(post))=NaN; spkxrun(cluspktimes<min(post))=NaN;
        spkyrun(cluspktimes>max(post))=NaN; spkyrun(cluspktimes<min(post))=NaN;
        spkspeed(cluspktimes>max(post))=NaN; spkspeed(cluspktimes<min(post))=NaN;
    end

end

function[errormessage,max_ampwaves, max_channelwaves, spk_widthwaves, ori] = make_spike_plots(separate_tetrodes, i, cluwaves, fig_rows, fig_cols, waveplotstile, cluspktimes, total_time, rasterplottile, refractoryperiodtile, thetacorrtile)
    %% spike plots - waveforms spike-time autocorrelograms and firing rate against time
    errormessage=strcat('making spike plots - ',char(separate_tetrodes+1),'cluster - ',num2str(i));
    [max_ampwaves,max_channelwaves,spk_widthwaves,ori]=plotwaveforms(cluwaves,[fig_rows fig_cols waveplotstile]);
    %[max_ampwhites,max_channelwhites,spk_widthwhites]=plotwaveforms(cluwhites,[fig_rows fig_cols whiteplotstile]);
    plotspikehist(cluspktimes,total_time,[fig_rows fig_cols rasterplottile]);
    plotspktimeautocor(cluspktimes,10,[fig_rows fig_cols refractoryperiodtile]);
    plotspktimeautocor(cluspktimes,250,[fig_rows fig_cols thetacorrtile]);
end

function[errormessage,skaggs,spars,cohe,max_firing,coverage,skaggsrun,sparsrun,coherun,max_firingrun,grid_score,grid_scorerun,frh_hd,meandir_hd,r_hd] = make_position_plots(OpenField,separate_tetrodes,i,posx,posy,spkx,spky,fig_rows,fig_cols,postile,pixel_ratio,post,ratemaptilerun,posmaptilerun,ratemaptile,posmaptile,posxrun,posyrun,spkxrun,spkyrun,postilerun,speed_cut,gridcortile,gridcortilerun,hdpolartile,speedtile,hd,spkhd,sampling_rate,cluspktimes,speed,spkspeed)
    if OpenField==1 % only do this if it's an open field session
        errormessage=strcat('making position plots - ',char(separate_tetrodes+1),'cluster - ',num2str(i));
        plotposition(posx,posy,spkx,spky,[fig_rows fig_cols postile]);
        subplot(fig_rows, fig_cols, postile);
        title({sprintf('All data ->')},'color','k');
        [frmap,~,skaggs,spars,cohe,max_firing,coverage]=plotratemap(posx,posy,spkx,spky,pixel_ratio,post,[fig_rows fig_cols ratemaptile], posmaptile);
        plotposition(posxrun,posyrun,spkxrun,spkyrun,[fig_rows fig_cols postilerun]);
        subplot(fig_rows, fig_cols, postilerun);
        title({sprintf('Running above %.1f cm/sec ->',speed_cut)},'color','k');
        [frmaprun,~,skaggsrun,sparsrun,coherun,max_firingrun,~]=plotratemap(posxrun,posyrun,spkxrun,spkyrun,pixel_ratio,post,[fig_rows fig_cols ratemaptilerun],posmaptilerun);
        if sum(isnan(spkx))<length(spkx)
            [grid_score,~,~,~,~]=plotgrid(frmap,[fig_rows fig_cols gridcortile]);
            [grid_scorerun,~,~,~,~]=plotgrid(frmaprun,[fig_rows fig_cols gridcortilerun]);
            %% hd plot
            [frh_hd,meandir_hd,r_hd]=plothd(hd,spkhd,sampling_rate,[fig_rows fig_cols hdpolartile]);
            %% speed plot
            [speedscore]=calcspeedscore(post,cluspktimes,speed);
            plotspeed(speed,spkspeed,sampling_rate,speedscore,[fig_rows fig_cols speedtile]);
        else % create empty variables for saving to data matrix
            frh_hd=NaN; meandir_hd=NaN; r_hd=NaN; skaggs=NaN; spars=NaN; cohe=NaN; max_firing=NaN; grid_score=NaN; skaggsrun=NaN; sparsrun=NaN; coherun=NaN; max_firingrun=NaN; grid_scorerun=NaN;
        end
    end
end

function[errormessage,lightscore_p,lightscore_I,lightlatency,percentresponse,lightscore_p2,lightscore_I2,lightlatency2,percentresponse2,lightscore_p3,lightscore_I3,lightlatency3,percentresponse3,lightscore_p4,lightscore_I4,lightlatency4,percentresponse4] = make_opto_plots(Opto,separate_tetrodes,i,cluspktimes,LEDons,LEDoffs,fig_rows,fig_cols,optotile,GSQ,cluwaves,optohisttile,optoplotstile,ori)
%% optoplots - waveforms, raster, and firing rate histogram
    if Opto==1 % only do this if it's an opto-tagging session
        errormessage=strcat('making opto plots - ',char(separate_tetrodes+1),'cluster - ',num2str(i));
        [onspikes]=plotoptoraster(cluspktimes,LEDons,LEDoffs,[fig_rows fig_cols optotile],GSQ);
        if length(LEDons)>174 && length(LEDons)<200 && GSQ==0
            [lightscore_p,lightscore_I,lightlatency,percentresponse]=plotoptohist(LEDons(1:100),LEDoffs(1:100),cluspktimes(cluspktimes<LEDons(101)),[fig_rows fig_cols optohisttile(3:4)]);
            [lightscore_p2,lightscore_I2,lightlatency2,percentresponse2]=plotoptohist(LEDons(101:125),LEDoffs(101:125),cluspktimes(cluspktimes<LEDons(126) & cluspktimes>LEDons(100)),[fig_rows fig_cols optohisttile(1)]);
            [lightscore_p3,lightscore_I3,lightlatency3,percentresponse3]=plotoptohist(LEDons(126:175),LEDoffs(126:175),cluspktimes(cluspktimes>LEDons(125)),[fig_rows fig_cols optohisttile(2)]);
        elseif length(LEDons)>200 && GSQ==0
            [lightscore_p,lightscore_I,lightlatency,percentresponse]=plotoptohist(LEDons(1:100),LEDoffs(1:100),cluspktimes(cluspktimes<LEDons(101)),[fig_rows fig_cols optohisttile(3)]);
            [lightscore_p2,lightscore_I2,lightlatency2,percentresponse2]=plotoptohist(LEDons(101:200),LEDoffs(101:200),cluspktimes(cluspktimes<LEDons(201)),[fig_rows fig_cols optohisttile(4)]);
            [lightscore_p3,lightscore_I3,lightlatency3,percentresponse3]=plotoptohist(LEDons(201:225),LEDoffs(201:225),cluspktimes(cluspktimes<LEDons(226) & cluspktimes>LEDons(200)),[fig_rows fig_cols optohisttile(1)]);
            [lightscore_p4,lightscore_I4,lightlatency4,percentresponse4]=plotoptohist(LEDons(226:275),LEDoffs(226:275),cluspktimes(cluspktimes>LEDons(225)),[fig_rows fig_cols optohisttile(2)]);

        else
            [lightscore_p,lightscore_I,lightlatency,percentresponse]=plotoptohist(LEDons,LEDoffs,cluspktimes,[fig_rows fig_cols optohisttile]);
        end
        if ~isempty(onspikes)
            lightwaves=cluwaves(:,:,onspikes);
            [~,~,~]=plotwaveforms(lightwaves,[fig_rows fig_cols optoplotstile],ori);
        end
    end
end

function[errormessage] = save_plots(separate_tetrodes,i,figure2,sessionid,tet,format)
    %% save plot
    errormessage=strcat('saving figure - ',char(separate_tetrodes+1),'cluster - ',num2str(i));
    id = [sessionid '-Tetrode-' num2str(tet) '-Cluster-' num2str(i)];
    annotation('textbox', [0.1, 1.0, 1.0, 0], 'string', id,'FontSize',20);
    saveas(figure2,fullfile(strcat('Figures',num2str(separate_tetrodes)),id),'fig');
    %saveas(figure2,strcat(outfile,'/Figures/',id),'fig');
    set(gcf,'PaperUnits','centimeters');
    set(gcf,'PaperPosition',[0 0 56.56 40]);
    saveas(figure2,fullfile(strcat('Figures',num2str(separate_tetrodes)),id),format);
    %saveas(figure2,strcat(outfile,'/Figures/',id),format);
    close(figure2);
end