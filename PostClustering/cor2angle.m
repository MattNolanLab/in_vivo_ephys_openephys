%%% Steven Huang<s.huang@ed.ac.uk> %%%
function angle = cor2angle(x,y)
angle = acosd(x);
for ii=1:length(y)
  if (y(ii)<0)
    angle(ii) = -angle(ii);
  end
end
