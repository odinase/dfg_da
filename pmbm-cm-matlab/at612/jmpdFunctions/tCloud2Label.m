function labels = tCloud2Label(tCloud)

% Function to obtain labels from particle sizes in JMPD
% Written by Edmund Brekke june 2007
% Corrected by Edmund Brekke december 2007
% @ tCloud:         Number of targets represented by each particle (level C)
% > labels:         Gives the particle that each A-level element belons to
% $ tCloud2EndInd

if(sum(tCloud < 0) > 0)
    error('tCloud cannot contain negative entries');
end

% Total number of alive elements

nElement = sum(tCloud);

% Need list of start indices for all particles (including dead ones)

endInd = tCloud2EndInd(tCloud);
begInd = endInd - tCloud + 1;
begInd = [begInd,nElement];

% Label each element so that we know which particle it belongs to

labels = zeros(1,sum(tCloud));
for p=1:length(tCloud)
    if(tCloud(p) ~= 0)
        labels(begInd(p):endInd(p)) = p;
    end
end