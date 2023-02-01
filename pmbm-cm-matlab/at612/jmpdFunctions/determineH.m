function [markCloud,nQC] = determineH(tCloud,xCloud,metric)

% Function to determine partitions in JMPD
% Written by Edmund Brekke during July 2007
% @ tCloud:     Particle cardinalities
% @ xCloud:     A-level particle cloud
% @ metric:     Metric to determine when two elements interact with respect to threshold 1
% @ tMax:       Upper bound on elements per particle
% > markCloud:  A-level partition assignments
% $ tCloud2BegInd

% Coding in determinePartitions() didnt work. Now only distinguish partitions.

nPart = size(tCloud,2);
nElts = sum(tCloud);
begInd = tCloud2BegInd(tCloud);
endInd = tCloud2EndInd(tCloud);
markCloud = zeros(1,nElts);

nQC = zeros(1,nPart);

for p=1:nPart
    if(tCloud(p) > 0)
        marks = partitionLinkage(xCloud(:,begInd(p):endInd(p)),metric);
        markCloud(begInd(p):endInd(p)) = marks(2,:);
        nQC(p) = length(unique(marks));
    end
end

% This gives us B-level marks inserted into A-level cloud