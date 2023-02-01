function tCloud = endInd2TCloud(endInd)

% Function to find lengths of each particle in JMPD given ending indices of level A element array
% Written by Edmund Brekke june 2007
% No modifications done so far.
% @ endInd:     Level C list of ending indices in level A 
% > tCloud:     Number of elements in each particle (level C)

% This function employs the ABC hiearchy:
% Level A:  Level of element cloud (xCloud)
% Level B:  Level inside each particle
% Level C:  Level of particle cloud (tCloud)

endInd = [0,endInd];
tCloud = diff(endInd);

if(~issorted(endInd))
   error('endInd is not sorted'); 
end