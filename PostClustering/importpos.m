%% importpos - Tizzy import's position data for OEdata %%%
function [posx,posy,hd,post] = importpos(posfile,dir_big)
%
%   post unit = seconds
%   dir_big = the direction of the big spot on the headstage(1 = big spot on right, 2 = big spot on left)
%
%   Copyright (C) 2004 Sturla Molden
%   Edited by Steven Huang
%   Edited by Matthijs van der Meer [added head direction, averaging two LEDs for tracking]

kalman_cutoff = 0.05; % cutoff percentage of available position data to activate kalman filter

[tracker,trackerparam] = importvideotracker(posfile);
% if (trackerparam.num_colours ~= 4)
%     cprintf('red','\tgetpos requires 4 colours in video tracker file.\n');
% end

post = zeros(trackerparam.num_pos_samples,1);

% enforce the increment of timestamps as matt suggested
for ii = 2:trackerparam.num_pos_samples
  post(ii) = post(ii-1) + 0.02;
end

posx = zeros(trackerparam.num_pos_samples,1);
posy = zeros(trackerparam.num_pos_samples,1);
hd = zeros(trackerparam.num_pos_samples,1);
dx = zeros(trackerparam.num_pos_samples,1);
dy = zeros(trackerparam.num_pos_samples,1);
temp = zeros(trackerparam.num_pos_samples,4);

lastsample = 0;
sampled = 0;
swapped = 0;
lasthd = NaN;
good_hd = 0;

lookahead = 16;

for ii = 1:trackerparam.num_pos_samples

  sampled = sampled + 1;

  temp(sampled,:) = [tracker(ii).xcoord tracker(ii).ycoord];
  bigx = temp(sampled,1);% 1 is big spot x
  bigy = temp(sampled,3); % 3 is big spot y
  smallx = temp(sampled,2);% 2 is small spot x
  smally = temp(sampled,4); % 4 is small spot y
  
  posx(sampled) = nanmean([bigx,smallx]) + trackerparam.window_min_x;
  posy(sampled) = nanmean([bigy,smally]) + trackerparam.window_min_y;
  dx = bigx - smallx; 
  dy = bigy - smally; % 3 is big spot, 4 is small spot


  if (~isnan(dx) && ~isnan(dy))
    good_hd = good_hd + 1;

    if (dx==0 && dy==0) % this means there is no sufficient information to calculate head direction
      hd_ang = NaN; 
    else
      if(dir_big == 1) % big spot on right
        hd_ang = acosd(dy/sqrt(dx^2 + dy^2));
        hd_ang = hd_ang*sign0(dx);
      elseif(dir_big == 2) % big spot on left
        hd_ang = acosd(dy/sqrt(dx^2 + dy^2));
        hd_ang = hd_ang*sign0(dx);
        if (hd_ang>0)
          hd_ang = hd_ang - 180;
        else
          hd_ang = hd_ang + 180;
        end
      end
    end
    hd(sampled) = hd_ang;
  else

    hd(sampled) = NaN;

  end

end % end of for loop over samples

cprintf('black','\t%d samples of %d (%.2f%%) have HD\n',good_hd,sampled,(good_hd/sampled)*100);


post = post(1:sampled);
posx = posx(1:sampled);
posy = posy(1:sampled);
hd = hd(1:sampled);

post = post - post(1);

posx = posx(1:end-1);
posy = posy(1:end-1);
post = post(1:end-1);
hd = hd(1:end-1);

goodpos = length(posx(~isnan(posx)));

cprintf('black','\t%d samples of %d (%.2f%%) have at least one position data\n',goodpos,sampled,(goodpos/sampled)*100);

%%% HD interpolation for missing data

%n = length(hd);
%missing = zeros(n,1);
%missing(isnan(hd)) = 1;
%hd_x = cosd(-hd+90);
%hd_y = sind(-hd+90);
%hd_x = interp1(post(~missing), hd_x(~missing), post);
%hd_y = interp1(post(~missing), hd_y(~missing), post);

%for i = 8:n-7

   %hd_x(i) = mean(hd_x(i-7:i+7));
   %hd_y(i) = mean(hd_y(i-7:i+7));

%end

%for i = 1:n

    %hd(i) = acosd(([0 1]*[hd_x(i) hd_y(i)]')/(sqrt(hd_x(i)^2 + hd_y(i)^2)));
    %hd(i) = hd(i)*sign0(hd_x(i));

%end


% removing the padding minimum continuim at the end that happens sometimes from interrupting recording
minposx = min(posx);
minposy = min(posy);

hd = hdswap(hd, 8500); % if rotation is > 8000 degrees/s, swapped it by 180 degrees
% only remove the continuous mins in the end so that to specifically target the padding problem
while 1
  if(posx(end) == minposx && posy(end) == minposy)
    posx = posx(1:end-1);
    posy = posy(1:end-1);
    hd = hd(1:end-1);
    post = post(1:end-1);
  else
    break;
  end
end

[posx, posy] = remove_jump_pos(posx, posy, post);

% if head direction data is not available, assign them all to 0
if (length(hd(isnan(hd))) == length(hd))
  hd(isnan(hd)) = 0;
  cprintf('black','\tWARNING: there is no head direction data from position recording\n');
end

if (goodpos / sampled > kalman_cutoff)
  [post,posx,posy] = trajectory_kalman_filter(posx,posy,post);
  hd = hd_interp(hd); %linear iterpolation of missing hd data
else
  cprintf('red','\tWARNING: Kalman pos filter disabled due to too much missing data\n');
end

end_ind = findstr(posfile, '.POS');
end_ind = end_ind -1;
base_name = posfile(1:end_ind);
fout = [base_name '.mypos'];
fid = fopen(fout, 'w', 'native');

if (fid<0)
  cprintf('red','\tCould not open %s for writing\n',fout);
end

pos_all =[posx posy hd post]';

fprintf(fid, '%f %f %f %f\n', pos_all);
fclose(fid);


function [posx, posy] = arena_config(posx,posy,arena)
switch arena
    case 'room0'
        centre = [433 256];
        conversion = 387;
    case 'room1'
        centre = [421 274];
        conversion = 403;
    case 'room2'
        centre = [404 288];
        conversion = 400;
    case 'room3'
        centre = [238 230];
        conversion = 396;
    case 'room4'
        centre = [394 366];
        conversion = 502;
    case 'room5'
        centre = [400 287];
        conversion = 459;
    case 'room6'
        centre = [419 286];
        conversion = 280;
end
posx = 100 * (posx - centre(1))/conversion;
posy = 100 * (centre(2) - posy)/conversion;



