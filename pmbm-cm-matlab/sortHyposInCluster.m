function [hyposNew,hyposCardNew,clustersNew,clustersCardNew,probabilities,probLogsNew] = sortHyposInCluster(hypos,hyposCard,clusters,clustersCard,probLogs)

% Function to arrange hypotheses sorted per cluster

hyposNew = zeros(size(hypos));
hyposCardNew = zeros(size(hyposCard));
clustersNew = zeros(size(clusters));
clustersCardNew = clustersCard;
clustersNewTemp = zeros(size(clusters));
cBegs = tCloud2BegInd(clustersCard);
cEnds = tCloud2EndInd(clustersCard);
probLogsNew = zeros(1,size(clustersCard,2));
for iC=1:size(clustersCard,2)
    hInC = clusters(cBegs(iC):cEnds(iC));
    pInC = probLogs(hInC);
    [temp,ix] = sort(pInC,'descend');
    hInCNew = hInC(ix);
    clustersNewTemp(cBegs(iC):cEnds(iC)) = hInCNew; 
    probLogsNew(cBegs(iC):cEnds(iC)) = temp;
end

% It remains to re-order hypos and give the hypotheses new identities

[hyposNew,hyposCardNew] = reOrderC(hypos,hyposCard,clustersNewTemp);
clustersNew = 1:size(clusters,2);


probabilities = zeros(1,0);

probabilitiesCell = probLogs2Probabilities(probLogsNew,clustersNew,clustersCardNew);
for ii=1:size(clustersCard,2)
   probabilities(cBegs(ii):cEnds(ii)) = probabilitiesCell{ii};
end

probLogsNew = probLogs(clustersNewTemp);