%%% Steven Huang<s.huang@ed.ac.uk> %%%

function [count timestamps type value] = read_key(filename)
bytes_per_timestamp = 4;
time_base = 96000;

[fid,msg] = fopen(filename,'r','ieee-be');
if (fid == -1)
  disp(sprintf('File Load Error (%i, %s)',fid,msg));
  disp(sprintf('Current Directory %s',pwd));
  disp(sprintf('Opening file %s',filename));
  fclose(fid);
  dot_ind = find(filename == '.');
  filename = [filename(1:dot_ind(end)) 'inp'];
  fid = fopen(fname, 'r');
end

if (fid < 0)
   error(sprintf('Could not open %s\n',filename)); 
end    

% read all bytes, look for 'data_start' and 'data_end'
fseek(fid,0,-1);
sresult = 0;
end_result = 0;
[bytebuffer, bytecount] = fread(fid,inf,'uint8');

for ii = 10:length(bytebuffer)
  if strcmp( char(bytebuffer((ii-7):ii))', 'timebase' )
    disp('Setting timebase...');
    timebase_index = ii+2;
    jj = ii+1;
    while (bytebuffer(jj)~='h')
      jj = jj + 1;
    end
    timebase_string = char(bytebuffer(timebase_index:jj-2))';
    time_base = str2num(timebase_string);
    
    disp(['Setting new timebase from default to ',num2str(time_base)]);
  end
  if strcmp( char(bytebuffer((ii-9):ii))', 'data_start' )
    sresult = 1;
    start_index = ii+1;
  end
  
  if strcmp(char(bytebuffer((ii-7):ii))', 'data_end' )
    end_result = 1;
    end_index = ii-10;
  %else
  %  end_result = 1;
  %  end_index = ii;
  end
end


if (~sresult)
    fclose(fid);
    error(sprintf('%s does not have a data_start marker', filename));
end

if (~end_result)
  fclose(fid);
  error(sprintf('%s does not have a data_end marker', filename));
end

big_endian_vector =  (256.^((bytes_per_timestamp-1):-1:0))';

data_size = (end_index - start_index + 1) / 7;

timestamps = [];
type = {};
value = {};
count = 1;

% each record is 7 bytes, first four bytes are big endian timestamp, 5th byte was keystroke type
% 6-7 bytes are keystroke values
for ii=start_index:7:end_index
  if strcmp(char(bytebuffer(ii:ii+6))', 'data_en')
    break;
  end
  timestamps = [timestamps, sum(bytebuffer(ii:ii+3).*big_endian_vector)];

  type{count} = char(bytebuffer(ii+4));

  if (type{count} == 'K')
    value{count} = char(bytebuffer(ii+5:ii+6));
  else
    value{count} = bytebuffer(ii+6);
  end  
  count = count + 1;
end


fclose(fid);
timestamps = timestamps / time_base;



