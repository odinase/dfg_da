function [contentionClusters,contentionTracks] = contentionCheck(pqI,bestLocal)

nCL = length(pqI.clustercontrib);

contentionClustersBool = false(1,nCL);
contentionTracks = zeros(1,0);
for j1=1:nCL
    for j2=(j1+1):nCL
        parentInJ1 = pqI.clustercontrib(j1).parentInBestList;
        parentInJ2 = pqI.clustercontrib(j2).parentInBestList;
        [sharedMea,ia1,ia2] = intersect(bestLocal(j1).clustercontrib(parentInJ1).meaIndOpt,bestLocal(j2).clustercontrib(parentInJ2).meaIndOpt);
        if(~isempty(sharedMea))
            contentionClustersBool(j1) = true;
            contentionClustersBool(j2) = true;
            t1 = bestLocal(j1).clustercontrib(parentInJ1).tracks(ia1);
            t2 = bestLocal(j2).clustercontrib(parentInJ2).tracks(ia2);
            contentionTracks = [contentionTracks,[t1,t2]];
        end
    end
end
contentionClusters = find(contentionClustersBool);
