function [post,posx,posy,hd,HDint]= GetPosSyncedAxona;

%% import position data
pname='*.POS';
pname=dir(pname);
pname=pname.name;

[posx,posy,hd,post] = importpos(pname,2);

%% import sync data
inpname='*.INP';
inpname=dir(inpname);
inpname=inpname.name;
[AXcount AXtimestamps AXtype AXeventdata] = read_key(inpname);

AXeventdata=cell2mat(AXeventdata);
BZpulse=AXtimestamps(AXeventdata==0);
BZpulse(1)=[];
%% Get Pulse data
try
[Ons,Offs]=GetSync;
OEpulse=(Ons+Offs)./2;
OEpulse=OEpulse';
catch
[~, ~, PosON, PosOFF]=GetPulse(3, [1 2]);
OEpulse=[PosON PosOFF];
end
if length(OEpulse)<2
  [~, ~, PosON, PosOFF]=GetPulse(3, [1 2]);
OEpulse=[PosON PosOFF];
end
OEpulse
%% check if there is a significant lag in one of the systems
if abs(length(OEpulse)-length(BZpulse))>0
    OEpulse=OEpulse(1:length(BZpulse));
end
try
BZnorm=BZpulse-min(BZpulse)+min(OEpulse);
lag=OEpulse-BZnorm;

if max(abs(lag))>2/30
    cprintf('red','You have a significant lag between systems')
    max(abs(lag))
    max(post)-min(post)
      %post = post+(min(OEpulse)-min(BZpulse));
      %% need to rewrite this to work if there is a lag
      L=BZpulse;
      W=diff(L);
          postnew = post+(min(OEpulse)-post(L(1))); 
    lag=OEpulse-postnew(L);
    postmorph=postnew;
    for j=2:length(L)
        inc=lag(j)/W(j-1);
        incs=1:W(j-1); incs=incs*inc;
        postmorph(L(j-1)+1:L(j))=postnew(L(j-1)+1:L(j))+incs';
        
    end
    lag2=OEpulse-postmorph(L);
    
    cprintf('green','The lag has been fixed')
    maxlag=max(lag2)
    if maxlag<2/30
        postnew=postmorph;
    end
else
    post = post+(min(OEpulse)-min(BZpulse));
end
catch
    cprintf('red', 'pulses do not match');
    OEpulse
    BZpulse
    diff(OEpulse)
    diff(BZpulse)
end
%% Syncronise data

total_time=max(post)-min(post);

HDint=hd;