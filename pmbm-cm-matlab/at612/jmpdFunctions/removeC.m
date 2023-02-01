function [indicesRemaining,tCloudNew] = removeC(c,tCloud)

% Function to simply remove partitions
% Written by Edmund Brekke during December 20007
%@c:                Partitions to be removed
%@tCloud:           Lengths of all partitions
%>indicesRemaining: A-level indices of elements in remaining partitions
%>tCloudNew:        Lengths of remaining partitions
% Notice that this function is a simpler version of pickIndC()

% The ABC-hierarchy
% level A: All partitions stacked along 2nd dimension
% level B: Inside each partition
% level C: Partitions as entities

if(max(c) > length(tCloud))
    error('Cannot access partition with higher C index than number of partitions');
end
if(sum(c < 0) > 0)
    error('Cannot access partitions with negative C index');
end

% Determine remaining partition lengths

tCloudNew = tCloud;
begInd = tCloud2BegInd(tCloud);
tCloudNew(c) = [];

% Mark A-level elements to be removed

removeLabels = zeros(size(tCloud));
for ii=1:length(c)
    indices=begInd(c(ii)):(begInd(c(ii))+tCloud(c(ii))-1);
    removeLabels(indices) = 1;
end

% Remove A-level elements

indicesRemaining = 1:sum(tCloud);
indicesRemaining(logical(removeLabels)) = [];

