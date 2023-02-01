function [trackSumProbs,trackTotalProbs,includedTracks] = trackProbGivenCluster(iCList,trackFile,inCol,hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHyposWill)

% New version of trackProbAccumulate that does not use meaHistCol
% Written by Edmund Brekke, starting 3rd of July 2020.
% New version that only returns probabilities in a list of given clusters

% First of all, identify included tracks in given clusters

nTracks = size(trackFile,2);
includedTracks = NaN*zeros(1,nTracks);
counterIT = 0;
for ii=1:length(iCList)
    c = iCList(ii);
    tracksInC = cluster2Tracks(c,hyposWill,hyposCardWill,clustersWill,clustersCardWill);
    includedTracks(counterIT+1:counterIT+size(tracksInC,2)) = tracksInC;
    counterIT = counterIT + size(tracksInC,2);
end
includedTracks(counterIT+1:end) = [];


reverseClusterMap = zeros(size(clustersWill));
for ii=1:length(iCList)
    reverseClusterMap(iCList(ii)) = ii; 
end




nTracksI = size(includedTracks,2);

%probabilitiesCell = probLogs2Probabilities(probLogHyposWill,clustersWill,clustersCardWill);
probabilitiesCell = probLogs2ProbabilitiesGivenCluster(probLogHyposWill,clustersWill,clustersCardWill,iCList);

trackSumProbs = zeros(1,nTracksI);
trackTotalProbs = zeros(1,nTracksI);
clusterPerTrack = zeros(1,nTracksI);
hyposPerTrack = cell(1,nTracksI);

for iT=1:nTracksI
    
    t = includedTracks(iT);
    %meaSeq = meaHistCol(:,t);
    %[tracks,hyposCol,clusterNumbers] = findTrack(meaSeq,meaHistCol,hyposWill,hyposCardWill,clustersWill,clustersCardWill);
    
    [clusterNumbers,hyposCol,probabilities] = track2Cluster(t,hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHyposWill);
    
    cni = reverseClusterMap(clusterNumbers);
    
    % Must use an alternative to find track
    
    
    if(length(unique(clusterNumbers)) > 1)
        error('several clusters assigned to one track');
    end
    clusterPerTrack(t) = unique(clusterNumbers);
    
    if(length(hyposCol) == 1)
        hyposPerTrack(t) = hyposCol;
    else
        hyposPerTrack{t} = cell2mat(hyposCol);
    end
        
      
    
    if(~isempty(clusterNumbers) && ~isnan(clusterNumbers(1)))
        
        % Need to convert a-level hypothesis numbers to b-level
        
        hList = hyposCol{1};
        inClustersA = [];
        for ii=1:size(hList,2)
            
            inClustersA = [inClustersA,find(clustersWill == hList(ii))];
        end
        
        [b,c] = a2bc(inClustersA,clustersCardWill);
        
        probabilitiesThis = probabilitiesCell{cni};
        trackSumProbs(t) = sum(probabilitiesThis(b));
        trackTotalProbs(t) = trackSumProbs(t)*trackFile(inCol.exi,t);
    end
end