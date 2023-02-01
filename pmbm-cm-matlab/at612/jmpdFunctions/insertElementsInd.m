function [indices1,indices2,tCloudNew] = insertElementsInd(particles,tCloud)

% Function to decide where to insert newborn particles in level A cloud in JMPD
% Written by Edmund Brekke during December 2007
% @particles:       Partitions which are going to get an additional element
% @tCloud:          Original lengths of all partitions
% >indices1:        Indices of the old elements in new level A cloud
% >indices2:        Indices of the new elements in new level A cloud
% >tCloudNew:       Lengths of all partitions after insertion
% $tCloud2EndInd

if(max(particles) > length(tCloud))
    error('Cannot access partition with higher C index than number of partitions');
end
if(sum(particles < 0) > 0)
    error('Cannot access partitions with negative C index');
end

% Obtain end indices for new cloud

tCloudNew = tCloud;
tCloudNew(particles) = tCloudNew(particles) + 1;
endInd = tCloud2EndInd(tCloudNew);

% List appropriate indices for old and new elements

indices2 = endInd(particles);
if(~isempty(endInd))
    indices1 = 1:endInd(end);
else
   indices1 = []; 
end
indices1(indices2) = [];