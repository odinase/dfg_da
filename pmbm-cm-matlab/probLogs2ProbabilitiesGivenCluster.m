function probabilitiesCell = probLogs2ProbabilitiesGivenCluster(probLogs,clusters,clustersCard,iCList)

% Revise probLogs2Probabilities function to only work in a limited selection of clusters





probabilitiesCell = cell(length(iCList),1);

begs = tCloud2BegInd(clustersCard);
ends = tCloud2EndInd(clustersCard);

for jj=1:length(iCList)
   
    ii= iCList(jj);
    a = exp(probLogs(clusters(begs(ii):ends(ii))));
    probabilitiesCell{jj} = a/sum(a);
    
end
