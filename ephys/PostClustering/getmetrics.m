function [cluid, bursting_parent,dur_sec,firing_rate,isolation,noise_overlap,num_events,overlap_cluster,peak_amp,peak_noise,peak_snr,t1_sec,t2_sec]=getmetrics(fname)
%% opens the cluster_metrics.json file and extracts all the metrics as a matrix of doubles
fid=fopen(fname);
metrics=fscanf(fid,'%s');
metrics=strsplit(metrics,','); % split file at commas
variables={'"bursting_parent":','"dur_sec":','"firing_rate":','"isolation":','"noise_overlap":','"num_events":','"overlap_cluster":','"peak_amp":','"peak_noise":','"peak_snr":','"t1_sec":','"t2_sec":'};
for i=1:length(variables)
    %% get the variable name
    variable=variables(i);
    varname=char(variable);
    %% find the lines coresponding to that variable and extract to output
index=strfind(metrics,variable);
index=~cellfun('isempty',index);
output=metrics(index);
%% trim to just the numbers
ind=strfind(output,varname);
ind=min(cell2mat(ind));
output = cellfun(@(x) x(ind+1:end), output, 'un', 0); 
ind=strfind(output,':');
ind=min(cell2mat(ind));
output = cellfun(@(x) x(ind+1:end), output, 'un', 0);
ind=strfind(output,'}');
ind=min(cell2mat(ind));
if ~isempty(ind)
output = cellfun(@(x) x(1:ind), output, 'un', 0);
end
%% convert cell array to matrix
output=cellfun(@str2double, output);
%% save as variable
ind=strfind(varname,'"');
varname=varname(ind(1)+1:ind(2)-1);
v=genvarname(varname);
eval([v '= output;']);
end
%% create cluster list variable
cluid=1:length(output);