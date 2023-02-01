function tracksInC = cluster2Tracks(c,hypos,hyposCard,clusters,clustersCard)

% Function to pick all tracks in a given cluster

begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
cHypos = clusters(begsC(c):endsC(c));
aInCluster = pickIndC(cHypos,hyposCard);
tracksInC = unique(hypos(aInCluster));


