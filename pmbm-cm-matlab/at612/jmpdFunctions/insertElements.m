function [xCloudNew,tCloudNew] = insertElements(particles,newBorn,xCloud,tCloud,xDim)

% Function to insert newborn particles in level A cloud in JMPD
% Written by Edmund Brekke during June 2007
% Modified by Edmund Brekke during December 2007
% @ particles:  Level C indices of particles to get newborns
% @ xCloud:     Level A cloud of elements
% @ newBorn:    Level A cloud of newborn elements
% @ tCloud:     Level C list of targets per particle
% > xCloudNew: xCloud and newBorn merged according to particles and tCloud
% $ insertElementsInd 
% $ tCloud2EndInd

if(size(particles,2) ~= size(newBorn,2))
    error('List of particles must have same size as list of newborn elements');
end
if(~isempty(newBorn) && size(newBorn,1) ~= xDim)
    error('Dimension mismatch between newBorn and xDim');
end
if(~isempty(xCloud) && size(xCloud,1) ~= xDim)
    error('Dimension mismatch between xCloud and xDim');
end

% Find indices in new A-level cloud where additional elements are to be inserted

[oldInd,insertInd,tCloudNew] = insertElementsInd(particles,tCloud);

% Convert indices to logical vector

insertMarks = false(1,sum(tCloudNew));
insertMarks(insertInd) = true;

% Insert newborn elements

xCloudNew = zeros(xDim,sum(tCloudNew));
xCloudNew(:,insertMarks) = newBorn;
if(~isempty(xCloud))
    xCloudNew(:,~insertMarks) = xCloud;
end