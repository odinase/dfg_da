function [clustersOut,clustersCardOut,designatedAfterPruning] = pruningAdjustClusters(clusters,clustersCard,toBeKept,toBeRemoved,designated,hyposPrune,hyposCardPrune)

nH = size(clusters,2);


temp2(toBeKept) = 1:length(toBeKept);
designatedAfterPruning = temp2(designated);

% Did I forget renumbering of hypothesis-pointers in clusters?

whereInClusters(clusters) = 1:size(clusters,2);

% Try again to construct clustersShifted

hypoRem = clusters(toBeRemoved); % Hypotheses to be removed in hypothesis-oriented numbering
increments = ones(1,nH);
increments(hypoRem) = 0;
newIndHypo = cumsum(increments); % New hypothesis numbers in hypothesis-oriented numbering

newIndHypo(clusters(toBeRemoved)) = 0;

clustersShifted = zeros(1,nH);
clustersShifted(whereInClusters) = newIndHypo;


if(any(clustersShifted(toBeRemoved)) > 0 && k==3)
    error('Zeros in clustersShifted do not conform with toBeRemoved');
end

% Get reversal of order of hypotheses in clusters

unsorted = 1:length(clusters);
newInd2(clusters) = unsorted;

% Remove cluster pointers to removed hypotheses

clustersPrune = clustersShifted(clustersShifted > 0);
[temp,clustersCardPrune] = removeIndA(toBeKept,clustersCard); 

if(max(clustersPrune) > length(hyposCardPrune))
    error('Cannot access partition prune');
end
[shareTracks,shareClusters] = testClustersShareTracks(clustersPrune,clustersCardPrune,hyposPrune,hyposCardPrune);
if(~isempty(shareTracks))
    error('tracks shared among clusters in pruning 1');
end

% I should also remove clusters which are not supported by hypotheses

supportedClusters = find(clustersCardPrune > 0);
[aInd,clustersCardOut] = pickIndC(supportedClusters,clustersCardPrune);
clustersOut = clustersPrune(aInd);

if(any(clustersOut == 0))
    error('Zero should not be in any cluster');
end

