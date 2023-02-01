function [trackSumProbs,trackTotalProbs] = trackProbAccumulate(meaHistCol,trackFile,inCol,hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHyposWill)

nTracks = size(meaHistCol,2);

probabilitiesCell = probLogs2Probabilities(probLogHyposWill,clustersWill,clustersCardWill);

trackSumProbs = zeros(1,nTracks);
trackTotalProbs = zeros(1,nTracks);
clusterPerTrack = zeros(1,nTracks);
hyposPerTrack = cell(1,nTracks);

for t=1:nTracks
    meaSeq = meaHistCol(:,t);
    [tracks,hyposCol,clusterNumbers] = findTrack(meaSeq,meaHistCol,hyposWill,hyposCardWill,clustersWill,clustersCardWill);
    
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
        
        probabilitiesThis = probabilitiesCell{clusterNumbers};
        trackSumProbs(t) = sum(probabilitiesThis(b));
        trackTotalProbs(t) = trackSumProbs(t)*trackFile(inCol.exi,t);
    end
end