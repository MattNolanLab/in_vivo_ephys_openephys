% linear interpolation of the missing head directions
% can only cope with very small number of missing values

function hd = hd_interp(hd)

hd1 = NaN;
ind1 = 0;
% if there is a leading NaN values. set them all to the first available hd value
while (isnan(hd1))
  ind1 = ind1+1;
  hd1 = hd(ind1);
end

for ii=1:ind1
  hd(ii) = hd(ind1);
end

for ii=ind1+1:length(hd)
  hd2 = hd(ii);
  ind2 = ii;
  if (~isnan(hd2))
    if ((ind2 - ind1)>1)
      hd_diff = hd2 - hd1;
      if (hd_diff > 180) % always take the smaller of the two possible answers as the correct one
        hd_diff = hd_diff - 360;
      elseif(hd_diff < -180)
        hd_diff = hd_diff + 360;
      end
      hd_incre = hd_diff / (ind2-ind1);
      for jj=ind1+1:ind2-1
        hd(jj) = hd(jj-1) + hd_incre;
      end
    end
    hd1 = hd2;
    ind1 = ind2;
  end
end

% set the trailing NaNs to the last available hd value
hd1 = NaN;
ind1 = length(hd) + 1;

while (isnan(hd1))
  ind1 = ind1 - 1;
  hd1 = hd(ind1);
end

if (ind1 < length(hd))
  for ii=ind1+1:length(hd)
    hd(ii) = hd(ind1);
  end
end

hd = cor2angle(cosd(hd), sind(hd));
