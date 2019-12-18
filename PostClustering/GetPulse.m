%PlotLight Script for plotting spikes against the light pulses from event
%data rather than continuous
function [pulseons, pulseoffs, PosON, PosOFF]=GetPulse(lightseq, posseq)

[data, timestamps, info] = load_open_ephys_data('all_channels.events');

pulses=timestamps(data==2);
syncs=timestamps(data==0);

syncs=syncs(3:2:length(syncs)-1);
if posseq
PosON=syncs(posseq(1));
PosOFF=syncs(posseq(2));
else
    PosON=[];
    PosOFF=[];
end
round(syncs)
% length(syncs)
if lightseq
cutoff=(min(syncs(lightseq))-1);
pulses=pulses(pulses>(cutoff-1));
ONInds=[1:2:length(pulses)-1];
OFFInds=[2:2:length(pulses)];
pulseons=pulses(ONInds);
pulseoffs=pulses(OFFInds);
else
    pulseons=[]; pulseoffs=[];
end
end

