function out = sign0(in)
%SIGN0  Signum function.
% like SIGN, but returns 1 if input is 0.

if (in < 0)
    out = -1;
else
    out = 1;
end

