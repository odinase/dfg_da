function [score,scorePerTrack,scorePerCluster] = evaluateAssignment(pq)

nCL = length(pq.clustercontrib);
scorePerCluster = zeros(1,nCL);

nTracks = size(pq.rewardMatrix,1);
scorePerTrack = zeros(1,nTracks);
score = 0;
for ii=1:nTracks
    
    item = find(pq.meaEntire == pq.meaEntireOpt(ii));
    
    rew = pq.rewardMatrix(ii,item);
    score = score + rew;
    scorePerTrack(ii) = rew;
    trackID = pq.tracksEntire(ii);
    for c=1:nCL
       if(ismember(trackID,pq.clustercontrib(c).tracks))
          scorePerCluster(c) =  scorePerCluster(c) + rew;
       end
    end
    
end
