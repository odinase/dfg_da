function [hyposNew,hyposCardNew,clustersOut,clustersCardOut,probLogHyposPrune,trackFile,trackFileShadow,meaHistCol,designatedAfterPruning,existences] = ...
    pruningPmbm(hypos,hyposCard,clusters,clustersCard,probLogHypos,trackFile,trackFileShadow,meaHistCol,designated,inCol,nHypoTotalMax,k)

% Revise the function entirely November 2018 since there is less need to update track collections in PMBM
% Still: I should remove tracks that have no hypothesis support. 

% @nHypoTotalMax: Maximal number of hypotheses per cluster (new revision April 2021)

existences = trackFile(inCol.exi,:);
nH = size(clusters,2);
[shareTracks,shareClusters] = testClustersShareTracks(clusters,clustersCard,hypos,hyposCard);
if(~isempty(shareTracks))
    error('tracks shared among clusters before pruning');
end
if(length(clusters) ~= length(hyposCard))
   error('N of clustered hypotheses do not conform with N hypotheses given by hyposCard');
end
exiThreshold = 0.0001;

pLOld = probLogHypos;

% New version - try to only sort inside individual clusters
% NOTICE: This means that existence uncertainty also should be utilized for pruning purposes.

cBegs = tCloud2BegInd(clustersCard);
cEnds = tCloud2EndInd(clustersCard);
toBeRemoved = false(1,sum(clustersCard));

hBegs = tCloud2BegInd(hyposCard);
hEnds = tCloud2EndInd(hyposCard);

for c=1:size(clustersCard,2)
   
    clusterC = clusters(1,cBegs(c):cEnds(c));  % Hypothesis indices of this cluster
    %pLC = probLogHypos(cBegs(c):cEnds(c)); % Logarithmic probabilities of these hypotheses
    
    if(~isempty(clusterC))
        pLC = probLogHypos(clusterC); % Logarithmic probabilities of these hypotheses
        [pLSorted,iPL] = sort(pLC,'descend');
        probs = exp(pLSorted - pLSorted(1));
        probs = probs/sum(probs);
        a=1-cumsum(probs);
        d = 1:size(a,2);
        toBeRemovedSorted = (a < 0.0006 & probs < 0.0006) | d > nHypoTotalMax;
        %nH = size(toBeRemovedSorted,2);

        
        unsorted = 1:length(probs);
        inverseSort = zeros(1,length(probs));
        inverseSort(iPL) = unsorted;
        toBeRemovedLocal = toBeRemovedSorted(inverseSort);
        
        

        
        % I should also look into existence-based pruning here
        % Has ben fixed now: MUST BE REVISED SO THAT IT DOESNT AUTOMATICALLY KILL EMPTY HYPOS
        
        for ii=1:size(clusterC,2)
            tracksInH = hypos(1,hBegs(clusterC(ii)):hEnds(clusterC(ii)));
            exiInH = existences(tracksInH);
            %toBeRemovedLocal(ii) = toBeRemovedLocal(ii) || (  all(exiInH < exiThreshold));
            toBeRemovedLocal(ii) = toBeRemovedLocal(ii) || ( ~isempty(exiInH) &&  all(exiInH < exiThreshold));
        end
        
           
        % Insert cluster-wise pruning boolean in global pruning boolean
        
        %toBeRemoved(cBegs(c):cEnds(c)) = toBeRemovedLocal;
        toBeRemoved(clusterC) = toBeRemovedLocal;
    end
    
end
%toBeRemoved(designated) = false;
[toBeKept] = find(~toBeRemoved);




[hyposIndPrune,hyposCardPrune] = pickIndC(sort(clusters(toBeKept)),hyposCard);

hyposPrune = hypos(:,hyposIndPrune);
probLogHyposPrune =probLogHypos(~toBeRemoved);


% I should really separate the cluster adjustment stuff in a separate function


[clustersOut,clustersCardOut,designatedAfterPruning] = pruningAdjustClusters(clusters,clustersCard,toBeKept,toBeRemoved,designated,hyposPrune,hyposCardPrune);


begsHP = tCloud2BegInd(hyposCardPrune);
endsHP = tCloud2EndInd(hyposCardPrune);

% temp2(toBeKept) = 1:length(toBeKept);
% designatedAfterPruning = temp2(designated);
% 
% % Did I forget renumbering of hypothesis-pointers in clusters?
% 
% whereInClusters(clusters) = 1:size(clusters,2);
% 
% % Try again to construct clustersShifted
% 
% hypoRem = clusters(toBeRemoved); % Hypotheses to be removed in hypothesis-oriented numbering
% increments = ones(1,nH);
% increments(hypoRem) = 0;
% newIndHypo = cumsum(increments); % New hypothesis numbers in hypothesis-oriented numbering
% 
% newIndHypo(clusters(toBeRemoved)) = 0;
% 
% clustersShifted = zeros(1,nH);
% clustersShifted(whereInClusters) = newIndHypo;
% 
% 
% if(any(clustersShifted(toBeRemoved)) > 0 && k==3)
%     error('Zeros in clustersShifted do not conform with toBeRemoved');
% end
% 
% % Get reversal of order of hypotheses in clusters
% 
% unsorted = 1:length(clusters);
% newInd2(clusters) = unsorted;
% 
% % Remove cluster pointers to removed hypotheses
% 
% clustersPrune = clustersShifted(clustersShifted > 0);
% [temp,clustersCardPrune] = removeIndA(toBeKept,clustersCard); 
% 
% if(max(clustersPrune) > length(hyposCardPrune))
%     error('Cannot access partition prune');
% end
% [shareTracks,shareClusters] = testClustersShareTracks(clustersPrune,clustersCardPrune,hyposPrune,hyposCardPrune);
% if(~isempty(shareTracks))
%     error('tracks shared among clusters in pruning 1');
% end
% 
% % I should also remove clusters which are not supported by hypotheses
% 
% supportedClusters = find(clustersCardPrune > 0);
% [aInd,clustersCardOut] = pickIndC(supportedClusters,clustersCardPrune);
% clustersOut = clustersPrune(aInd);
% 
% if(any(clustersOut == 0))
%     error('Zero should not be in any cluster');
% end

% Finally, remove tracks that are unclaimed

hyposBegs = tCloud2BegInd(hyposCardPrune);
hyposEnds = tCloud2EndInd(hyposCardPrune);

claimedTrackBool = false(1,size(trackFile,2));
nHNew = size(hyposCardPrune,2);
for iH=1:nHNew
    h = hyposPrune(:,hyposBegs(iH):hyposEnds(iH));
    claimedTrackBool(h) = true;
end
tracksKeep = find(claimedTrackBool);
newForEachOld = zeros(1,size(trackFile,2));
newForEachOld(tracksKeep) = 1:size(tracksKeep,2);
hyposNew = newForEachOld(hyposPrune);
hyposCardNew = hyposCardPrune; % Relabeling track-pointers does not affect the hypothesis cardinalities

% Shouldnt I also update trackFile and meaHistCol here ??????


trackFile = trackFile(:,tracksKeep);
trackFileShadow = trackFileShadow(:,tracksKeep,:);
meaHistCol = meaHistCol(:,tracksKeep);


[shareTracks,shareClusters] = testClustersShareTracks(clustersOut,clustersCardOut,hyposNew,hyposCardNew);

if(~isempty(shareTracks))
    error('tracks shared among clusters in pruning 2');
end

% At this stage I should also remove empty hypotheses for all cases wheen I
% only have two hypotheses per cluster

existences = trackFile(inCol.exi,:);
twoSome = find(clustersCardOut == 2);
emptyHyposInd = find(hyposCardNew == 0);


[~,clustersWithEmpty] = a2bc(emptyHyposInd,clustersCardOut);
candSimplify = intersect(twoSome,clustersWithEmpty);

% So what do I do with these clusters?


if(~isempty(candSimplify))
    
    toBeRemoved = false(1,sum(clustersCardOut));
    
    
    begsH = tCloud2BegInd(hyposCardNew);
    endsH = tCloud2EndInd(hyposCardNew);
    begsC = tCloud2BegInd(clustersCardOut);
    endsC = tCloud2EndInd(clustersCardOut);
    
    for c=1:length(candSimplify)
        
        hC = clustersOut(begsC(candSimplify(c)):endsC(candSimplify(c)));
        [~,zeroCardH] = min(hyposCardNew(hC));
        pL = probLogHyposPrune(hC);
        probs = exp(pL-max(pL));
        probs = probs/sum(probs);
        onetwo = 1:2;
        otherH = setdiff(onetwo,zeroCardH);
        if(size(otherH,2) ~= 1)
           error('something went wrong'); 
        end
        tracksInOther = hyposNew(begsH(hC(otherH)):endsH(hC(otherH)));
        exi = existences(tracksInOther);
        
        exiNew = exi*probs(otherH);
        
%         if(k==8)
%            disp('show the crucial stuff');
%             exi
%             exiNew
%             probs(otherH)
%         end
        existences(tracksInOther) = exiNew;  
        
        
        pLNew = 0; % Can be any number.
        
        toBeRemoved(hC(zeroCardH)) = true;
    end
    
    [toBeKept] = find(~toBeRemoved);
    [hyposIndPrune,hyposCardPrune] = pickIndC(sort(clustersOut(toBeKept)),hyposCardNew);
    hyposPrune = hyposNew(:,hyposIndPrune);
    probLogHyposPrune =probLogHyposPrune(~toBeRemoved);
    
    % Also need to update the clusters
    
    [clustersOut,clustersCardOut,designatedAfterPruning] = pruningAdjustClusters(clustersOut,clustersCardOut,toBeKept,toBeRemoved,designatedAfterPruning,hyposPrune,hyposCardPrune);
    hyposNew = hyposPrune;
    hyposCardNew = hyposCardPrune;
    trackFile(inCol.exi,:) = existences;
end



 