function [hyposCell,topHyposSel] = displayTopHypotheses(meaColHist,hypos,hyposCard,clusters,clustersCard,probLogHypos,c,n)

% @c: The cluster number to be investigated
% @n: The number of topmost hypotheses that I want to display

cBegs = tCloud2BegInd(clustersCard);
cEnds = tCloud2EndInd(clustersCard);
hBegs = tCloud2BegInd(hyposCard);
hEnds = tCloud2EndInd(hyposCard);

clusterC = clusters(cBegs(c):cEnds(c));

probabilitiesCell = probLogs2Probabilities(probLogHypos,clusters,clustersCard);
probabilitiesThis = probabilitiesCell{c};

probLogsC = probLogHypos(clusterC);
[temp,ix] = sort(probLogsC,'descend');  % ix would give me b-level indices of hypotheses in clusters
topHypos =  clusterC(ix);


nMax = min(length(ix),n);

hyposCell = cell(3,nMax);

for ii=1:nMax
    h = topHypos(ii);
    aTracks = hypos(pickIndC(h,hyposCard));
    hyposCell{1,ii} = meaColHist(:,aTracks);
    hyposCell{2,ii} = aTracks;
    hyposCell{3,ii} = probabilitiesThis(ix(ii));
end

topHyposSel = topHypos(1:nMax);