function begInd = tCloud2BegInd(tCloud)

% Function to obtain beginning indices of each partition given number of elements in each partition
% Written by Edmund Brekke during December 2007
% @ tCloud:     Number of elements in each partition
% > begInd:     Beginning index of each partition


if(sum(tCloud < 0) > 0)
    error('tCloud cannot contain negative entries');
end

endInd = cumsum(tCloud);
begInd = endInd - tCloud + 1;