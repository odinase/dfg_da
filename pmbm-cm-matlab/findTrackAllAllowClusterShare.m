function [tracks,hyposCol,clusterNumbers] = findTrackAllAllowClusterShare(meaSeq,meaHistCol,hypos,hyposCard,clusters,clustersCard)

% Function to identify hypotheses and cluster which contain a track as given by a measurement sequence.
% Modified by Edmund Brekke in March 2021 to find all tracks that conform with non-NaN values in meaSeq
% @meaSeq:  Sequence of measurements ending at same time step as meaHistCol ends


% First check whether the given track fits into  meaHistCol

nT = size(meaHistCol,2);
nKM = size(meaHistCol,1);
nKT = size(meaSeq,1);
remainingK = nKM-nKT;

if(remainingK < 0)
   
    meaSeq(1:abs(remainingK),:) = [];
    remainingK = 0;
    
    %error('Candidate track should not have more elements than tracks in track file'); 
end

testK = (remainingK+1):nKM;
meaHistLimited = meaHistCol(testK,:);
meaHistLimited(isnan(meaHistLimited)) = -1;

meaSeqExt = repmat(meaSeq,[1,nT]);

meaHistLimited(isnan(meaSeqExt)) = -1;  % This line makes findTrackAll differ from the original findTrack function

meaSeqExt(isnan(meaSeqExt)) = -1;



%testVar = all(meaSeqExt == meaHistLimited | meaSeqExt == -1,1)
testVar = all(meaSeqExt == meaHistLimited ,1);
tracks = find(testVar);

hyposCol = zeros(1,0);
clusterNumbers = zeros(1,0);

for ii=1:size(tracks,2)
    [clustersI,hyposIA] = track2ClusterAllowClusterShare(tracks(ii),hypos,hyposCard,clusters,clustersCard);
    
    hyposCol = [hyposCol,hyposIA];
    clusterNumbers = [clusterNumbers,clustersI];
end


%[clusterNumbers,hyposCol] = track2Cluster(tracks,hypos,hyposCard,clusters,clustersCard);


