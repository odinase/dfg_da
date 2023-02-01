function tCloud = begInd2TCloud(begInd,nElement)

% Function to find lengths of each particle in JMPD given start indices of level A element array
% Written by Edmund Brekke during June 2007
% Modified by Edmund Brekke during December 2007
% Modified by Edmund Brekke during July 2019
% @ begInd:     Beginning index (at level A)of each partition
% @ nElement:   Total number of elements in full list (at level A)
% > tCloud:     Number of elements in each particle (level C)

% This function employs the ABC hiearchy:
% Level A:  Level of element cloud (xCloud)
% Level B:  Level inside each partition
% Level C:  Level of partition cloud (tCloud)

if(max(begInd > nElement + 1))
    error('Highest begInd cannot be higher than number of elements plus one');
end
if(~issorted(begInd))
   error('begInd is not sorted'); 
end

% Partition lengths are found by differencing the beginning indices

if(~isempty(begInd) && nElement > 0)
    begInd = [begInd,nElement+1];
    tCloud = diff(begInd);
    if(tCloud(end) == 0)
        tCloud(end) = 1;
    end
    if(tCloud(end) < 0)
        tCloud(end) = 0;
    end
elseif(nElement == 0)
    tCloud = [];
else
    error('If begInd is empty there cannot be more than zero elements.');
end