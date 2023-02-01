function [hyposNew,hyposCardNew,clustersNew,clustersCardNew,probLogsNew,new2OldTrackNumbers,old2NewTrackNumbers] = removeHypos(hRemove,hypos,hyposCard,clusters,clustersCard,probLogs,nT)

% Function to remove a collection of whole hypotheses
% Written by Edmund Brekke during December 2021

% Stuff from previous version of nScanMerge:
%         [aRemove,tCloudRemove,aRemain,tCloudRemain] = pickIndC(clustersToPrune,clustersCardNew);
%         [hyposRemaining1,clustersCardPostPrune] = removeC(clustersToPrune,clustersCardNew); % So far all looks good
%         [hEntriesRemaining,hyposCardPostPrune] = removeC(aRemove,hyposCardNew); % hyposCardPostPrune also looks sensible


% Construct new hypos and clusters without re-indexing due to removals

[hEntriesRemove,hCardRemove,hEntriesRemain,hyposCardNew] = pickIndC(hRemove,hyposCard); 
hyposAR = hypos(hEntriesRemain);
tracksFullyRemoved = setdiff(1:nT,hyposAR);
[~,~,cEntriesRemain,clustersCardNew] = removeIndA(hRemove,clustersCard);
clustersNew = 1:sum(clustersCardNew); % Re-indexing of clusters (i.e. hypothesis numbers) is trivial if no whole clusters are removed
probLogsNew = probLogs(cEntriesRemain);

% Re-indexing of hypos (i.e. of the track collection)

new2OldTrackNumbers = setdiff(1:nT,tracksFullyRemoved); % Mapping from new track numbers to the old track numbers
old2NewTrackNumbers = 1:numel(new2OldTrackNumbers); 
old2NewTrackNumbers(new2OldTrackNumbers) = old2NewTrackNumbers; % This should be the mapping from old track numbers to new track numbers
hyposNew = old2NewTrackNumbers(hyposAR);
if(any(hyposNew == 0))
    error('Zero elements from tracksNewNumbers should not be allowed to enter hyposNew');  
end

% If a whole cluster is removed: Just remove its cardinality which then will be zero.

clustersCardNew(clustersCardNew == 0) = [];


