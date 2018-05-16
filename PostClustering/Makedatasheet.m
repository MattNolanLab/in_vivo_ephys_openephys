%% make overall datasheet per animal
% should be run from the animal root folder i.e. Cohort4/U
% data output will be saved in the animal root folder

%first_part_of_spreadsheet = 27;

%% make output files and header row
dataoutnames={'data_all.csv', 'data_separate.csv'};
for i=1:length(dataoutnames)
    dataoutname=char(dataoutnames(i));
fid = fopen(dataoutname, 'w' );
header={'id','animal','day','tetrode','cluster','nspikes','coverage','avgFR', 'maxamplitude', 'maxchannel','spkewidth','HD_maxFR','meanHD', 'r_HD', 'skaggs','sparsity','spatialcoherence','maxFRspatial','gridscore','skaggsrun','sparsrun','spatialcoherencerun','maxFRspatialrun','gridscorerun','lightscoreP','lightscoreI','lightlatency','percentresponse','lightscore_p2','lightscore_I2','lightlatency2','percentresponse2','lightscore_p3','lightscore_I3','lightlatency3','percentresponse3','lightscore_p4','lightscore_I4','lightlatency4','percentresponse4','cluster' ,'goodcluster', 'firing_rate', 'FRpass', 'isolation', 'isolationpass' , 'noiseoverlap','noiseoverlappass','peakSNR' ,'peakSNRpass','burstingparent'};
incoming_header_length = 48;
fprintf(fid,'%s,',header{:});
fprintf(fid,'\n');
fclose( fid );
end
%% make crashlist file
fid=fopen('Makedatasheetstatus.csv','a');
fprintf(fid,'\n%s,', 'Foldername');
fprintf(fid,'%s,', 'Status');
fclose(fid);
%% find all openfield subfolders
contents=dir('*of'); 
dirflags=[contents.isdir];
foldernames=contents(dirflags);
%% loop through all  folders
messages={'collected data', 'no metric data present', 'unidentified error'};
for fold=1:length(foldernames)
    output=fold;
    foldername=char(foldernames(fold).name);
disp(foldername);
cd(foldername);
%% identify if there are files that need curating and haven't been yet
try
if exist('clustermetrics','dir')
    Curation % this adds curation related info to all_data and separate_data csv files
    message=1;
    data1 = csvread('datasave_all.csv');
    data2 = csvread('datasave_separate.csv');
    
    cd ..
    ind=strfind(foldername,'_');
    animal=foldername(1:ind(1)-1);
    date=foldername(ind(1)+1:ind(2)-1);
%     dlmwrite('data_all.csv',data,'delimiter',',','-append');
%     dlmwrite('data_separate.csv',data2,'delimiter',',','-append');
    
    for i=1:length(dataoutnames)
        dataoutname=char(dataoutnames(i));
        fid = fopen(dataoutname, 'a' );
        data=eval(strcat('data',num2str(i)));
    for r=1:size(data,1)
        fprintf(fid,'%s,%s,%s',foldername,animal,date);
        fprintf(fid,'%d,',data(r,:));
        fprintf(fid,'\n');
    end
        fclose( fid );
    end 
else
    message=2;
    cd ..
end
catch
    message=3; 
    if strfind(pwd,foldername)~=0
    cd ..
    end
end
disp(string(messages(message)));
fid=fopen('Curation2Status.csv','a');
fprintf(fid,'\n%s,', string(foldernames(output).name));
fprintf(fid,'%s,', string(messages(message)));
fclose(fid);
end