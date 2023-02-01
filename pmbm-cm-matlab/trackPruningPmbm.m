function [hyposNew,trackFile,trackFileShadow,meaHistCol] = trackPruningPmbm(hypos,trackFile,trackFileShadow,meaHistCol,k)

nT = size(trackFile,2);

% In this function, the goal is to remove all tracks that don't have hypothesis support. 
% This means that numbers on hypos will change, but its structure will not change. 
% Furthermore, hyposCard, clusters, clustersCard and probLogHypos will not change. 

%pruneTracks = setdiff(1:nT,hypos);
supportedTracks = intersect(1:nT,hypos);
nTNew = size(supportedTracks,2);

newTrackNumbers = 1:nTNew;
ntnInOld = NaN*zeros(1,nT);
ntnInOld(supportedTracks) = newTrackNumbers;


hyposNew = ntnInOld(hypos);


meaHistCol = meaHistCol(:,supportedTracks);
trackFile = trackFile(:,supportedTracks);
trackFileShadow = trackFileShadow(:,supportedTracks,:);



