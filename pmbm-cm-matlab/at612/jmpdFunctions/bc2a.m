function a = bc2a(b,c,tCloud)

% Function to determine A level indices from B&C level indices in JMPD
% Written by Edmund Brekke during December 2007
% @tCloud:  Partition lengths
% @b:       B-level indices (location in partition)
% @c:       C-leve indices (partition number)
% >a:       A-level indices (location in complete list)
% $tCloud2BegInd

% Add location in partition to beginning of partition

begInd = tCloud2BegInd(tCloud);
a = begInd(c) + b - 1;

if(sum(b > tCloud(c)) > 0)
    error('b exceeds corresponding limit in tCloud'); 
end
if((sum(b <= 0) > 0 || sum(c <= 0) > 0))
    error('B and C level indices must be positive. (However, they could be NaN)');
end