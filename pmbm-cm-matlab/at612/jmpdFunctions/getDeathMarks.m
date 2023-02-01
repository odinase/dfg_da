function deathMarks = getDeathMarks(levelAInd,tCloud)

% Function to mark elements for death on level A in JMPD
% Written by Edmund Brekke june 2007
% No modifications done so far.
% @ levelAInd:      Indices of elements to kill (level A)
% @ tCloud:         Number of targets represented by each particle (level C)
% > deathMarks:     Number of elements to be killed for each particle (level C)
% $ tCloud2EndInd
% $ tCloud2Label

% This function employs the ABC hiearchy:
% Level A:  Level of element cloud (xCloud)
% Level B:  Level inside each particle
% Level C:  Level of particle cloud (tCloud)

if(max(levelAInd) > sum(tCloud))
    error('Cannot extract higher A-level index than sum of partition lengths');
end
if(sum(levelAInd <= 0) > 0)
    error('A-level indices must be positive');
end

% Get number of particles

nPart = size(tCloud,2);

% Put labels on elements

labels = tCloud2Label(tCloud);

% Count killed elements for affected particles

deathMarks = zeros(1,nPart);
for t=1:size(levelAInd,2)
    deathMarks(labels(levelAInd(t))) = deathMarks(labels(levelAInd(t))) + 1;
end