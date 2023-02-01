function probabilitiesCell = probLogs2Probabilities(probLogs,clusters,clustersCard)

probabilitiesCell = cell(length(clustersCard),1);

begs = tCloud2BegInd(clustersCard);
ends = tCloud2EndInd(clustersCard);

for ii=1:length(clustersCard)
   
    a = exp(probLogs(clusters(begs(ii):ends(ii))));
    probabilitiesCell{ii} = a/sum(a);
    
end
