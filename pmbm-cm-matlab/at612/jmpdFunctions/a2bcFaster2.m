function [b,c] = a2bcFaster2(a,tCloud)

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

tCumsum = cumsum(tCloud);
c = zeros(size(a));

for ii=1:length(a)
   aTest = tCumsum < a(ii); 
   c(ii) = sum(aTest)+1; 
end

tCumsumExt = [tCumsum,tCumsum(end)];
tCloudExt = [tCloud,0];
subtrahendA = tCumsumExt(c) - tCloudExt(c);
b = a - subtrahendA;
c(isnan(a)) = NaN;
