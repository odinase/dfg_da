function endInd = tCloud2EndInd(tCloud)

% Function to obtain end indices of each partition given the number ofcelements in each partition
% Written by Edmund Brekke December 2007
% @ tCloud:     Number of elements in each partition
% > endInd:     End index of each partition

if(sum(tCloud < 0) > 0)
    error('tCloud cannot contain negative entries');
end

endInd = cumsum(tCloud);