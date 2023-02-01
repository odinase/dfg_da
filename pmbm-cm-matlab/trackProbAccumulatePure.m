function [trackSumProbs,trackTotalProbs,probabilitiesCell] = trackProbAccumulatePure(trackFile,inCol,hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHyposWill)

% New version of trackProbAccumulate that does not use meaHistCol
% Written by Edmund Brekke, starting 3rd of July 2020.


nTracks = size(trackFile,2);

probabilitiesCell = probLogs2Probabilities(probLogHyposWill,clustersWill,clustersCardWill);

trackSumProbs = zeros(1,nTracks);
trackTotalProbs = zeros(1,nTracks);
clusterPerTrack = zeros(1,nTracks);
hyposPerTrack = cell(1,nTracks);

for t=1:nTracks
    %meaSeq = meaHistCol(:,t);
    %[tracks,hyposCol,clusterNumbers] = findTrack(meaSeq,meaHistCol,hyposWill,hyposCardWill,clustersWill,clustersCardWill);
    
    [clusterNumbers,hyposCol,probabilities] = track2Cluster(t,hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHyposWill);
    
    
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
        
        [b,c] = a2bcFaster(inClustersA,clustersCardWill);
        
        probabilitiesThis = probabilitiesCell{clusterNumbers};
        trackSumProbs(t) = sum(probabilitiesThis(b));
        trackTotalProbs(t) = trackSumProbs(t)*trackFile(inCol.exi,t);
    end
end