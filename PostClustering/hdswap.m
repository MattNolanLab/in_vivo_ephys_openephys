% convert head direction to angular head velocity
% the time between the data points of the head direction vector is expected to be 20ms
function new_hd= hdswap(hd, max_ahv)
% max_ahv = maximum possible angular head velocity in degrees per second (170 degree / 0.02 = 8500)

ahv = zeros(1, length(hd) - 1); % the ahv vector will be 1 shorter
new_hd = hd;

swap = 1;

while swap
  swap = 0;
  hd1 = new_hd(1);
  ind1 = 1;
  for e=2:length(new_hd)-1
    hd2 = new_hd(e);
    ind2 = e;
    if (~isnan(hd2))
      hd_diff = hd2 - hd1;
      if (hd_diff > 180) % always take the smaller of the two possible answers as the correct one
        hd_diff = hd_diff - 360;
      elseif(hd_diff < -180)
        hd_diff = hd_diff + 360;
      end
      v = hd_diff / (0.02*(ind2-ind1));
      if(abs(v) > max_ahv) % this might be due to tracker got inverted, try reverse them
        new_hd2 = nan;
        if(hd2>0)
          new_hd2 = hd2 - 180;
        else
          new_hd2 = hd2 +180;
        end
        new_hd(e) = new_hd2;
        swap = swap +1;
      end
      hd1 = new_hd(e);
      ind1 = e;
    end
  end
  swap;
end
