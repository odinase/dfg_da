function [b,c] = a2bc(a,tCloud)

% Function to determine B&C level indices from A level indices in JMPD
% Written by Edmund Brekke during December 2007
% @a:       A-level indices (location in complete list)
% @tCloud:  Partition lengths
% >b:       B-level indices (location in partition)
% >c:       C-leve indices (partition number)
% $tCloud2BegInd
% $tCloud2EndInd



if(max(a) > sum(tCloud))
    error('Cannot extract higher A-level index than sum of partition lengths');
end
if(sum(a <= 0) > 0)
    error('A-level indices must be positive');
end

% Find beginning and end index of each partition

begInd = tCloud2BegInd(tCloud);
endInd = tCloud2EndInd(tCloud);

% Declare vectors to keep output indices

b = zeros(size(a));
c = zeros(size(a));

% Strategy: For each partition check whether each A-level index is inside it

for ii=1:length(begInd)
    for jj=1:length(a)
        if(a(jj) >= begInd(ii) && a(jj) <= endInd(ii) && tCloud(ii) > 0)           
            c(jj) = ii;
            b(jj) = a(jj) - begInd(ii) + 1; 
        end
    end
end