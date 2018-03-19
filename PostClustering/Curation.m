% Curationscript
% "firing_rate":  > 0.5
% "isolation":    > 0.95
% "noise_overlap":< 0.03
% "peak_snr":     > 1.5
%% set up threshold variables
frthresh=0.5;
isolationthresh=0.9; % 0.95
noisethresh=0.05; %0.03
snrthresh=1; %1.5

%% filenames for datasave
datanames={'datasave_all','datasave_separate'};

for separate_tetrodes=0:1;
    if separate_tetrodes==0
        fname='clustermetrics/all_cluster_metrics.json';
        if exist(fname,'file')
        [cluid, bursting_parent,dur_sec,firing_rate,isolation,noise_overlap,num_events,overlap_cluster,peak_amp,peak_noise,peak_snr,t1_sec,t2_sec]=getmetrics(fname);
        end
    else
        fnames={'T1_cluster_metrics.json', 'T2_cluster_metrics.json', 'T3_cluster_metrics.json', 'T4_cluster_metrics.json'};
        cluid=[];  firing_rate=[];isolation=[]; noise_overlap=[]; peak_snr=[]; bursting_parent=[];
        for f=1:length(fnames)
            %% get metrics data
            fname=strcat('clustermetrics/',fnames(f));
            fname=char(fname);
            if exist(fname,'file')
                [cluidi, bursting_parenti,dur_seci,firing_ratei,isolationi,noise_overlapi,num_eventsi,overlap_clusteri,peak_ampi,peak_noisei,peak_snri,t1_seci,t2_seci]=getmetrics(fname);
                cluid=[cluid cluidi];  firing_rate=[firing_rate firing_ratei]; isolation=[isolation isolationi]; noise_overlap=[noise_overlap noise_overlapi]; peak_snr=[peak_snr peak_snri]; bursting_parent=[bursting_parent bursting_parenti];
            end
        end
    end
    %% do curation based on thresholds specified at top of script
    cluid2=1:length(cluid);
    goodcluster=ones(size(cluid));
    goodcluster(firing_rate<frthresh)=0; 
    goodcluster(isolation<isolationthresh)=0; 
    goodcluster(noise_overlap>noisethresh)=0; 
    goodcluster(peak_snr<snrthresh)=0; 
    test=[cluid2' goodcluster' firing_rate' (firing_rate>frthresh)' isolation' (isolation>isolationthresh)' noise_overlap' (noise_overlap<noisethresh)' peak_snr' (peak_snr>snrthresh)' bursting_parent'];
    %% save curation and metric information along with original datasave data to a csv file
    dataname=char(strcat(datanames(separate_tetrodes+1),'.mat'));
    load(dataname);
    datasave=datasave(1:size(test,1),:);
    newdata=[datasave test];
    datafile=char(strcat(datanames(separate_tetrodes+1),'.csv'));
    csvwrite(datafile,newdata);
    clear datasave
    
    %% move "good"  and "bad" clusters to separate folders 
    if separate_tetrodes==0
        figfolder='SortingFigures_all';
    else
        figfolder='SortingFigures_separate';
    end
    if ~exist(strcat(figfolder,'_Curated'),'dir')
        mkdir(strcat(figfolder,'_Curated'));
        mkdir(strcat(figfolder,'_Rejects'));
    end
%         figurelist=dir(strcat(figfolder,'_PNG'));
%         name=char(figurelist(end).name);
%         ind=strfind(name,'Tetrode-'); 
%         name=name(1:ind-1);
        
        for i=1:length(goodcluster)
            figname=dir(strcat(figfolder,'_PNG/*-Cluster-',num2str(i),'.png'));
            figname=char(figname.name);
            if goodcluster(i)==1
                copyfile(strcat(figfolder,'_PNG/',figname),strcat(figfolder,'_Curated/',figname));
            else
                copyfile(strcat(figfolder,'_PNG/',figname),strcat(figfolder,'_Rejects/',figname));
            end
        end

    
    
end