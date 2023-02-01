function [hyposNew,hyposCardNew,clustersOut,clustersCardOut,probLogHyposPrune,trackFile,trackFileShadow,meaHistCol,existences] = ...
    pruningPmbmBid(hypos,hyposCard,clusters,clustersCard,probLogHypos,trackFile,trackFileShadow,meaHistCol,inCol,nHypoTotalMax,k)

% Revise the function entirely November 2018 since there is less need to update track collections in PMBM
% Still: I should remove tracks that have no hypothesis support. 
% Revision 2021: The Bid version of the pruning function should allow
% clusters to accumulate more hypotheses up to an upper limit. 


% @nHypoTotalMax: Maximal number of hypotheses per cluster (new revision April 2021)
% In Bid version nHypoTotalMax should actually be the maximum number of TOTAL hypotheses.

% -------------------------------------------------------------------------
% Bid version: I really should sort hypotheses before doing anything so that it is done once and for all


[hypos,hyposCard,clusters,clustersCard,probabilities,probLogHypos] = sortHyposInCluster(hypos,hyposCard,clusters,clustersCard,probLogHypos);

% -------------------------------------------------------------------------


existences = trackFile(inCol.exi,:);
nH = size(clusters,2);
[shareTracks,shareClusters] = testClustersShareTracks(clusters,clustersCard,hypos,hyposCard);
if(~isempty(shareTracks))
    error('tracks shared among clusters before pruning');
end
if(length(clusters) ~= length(hyposCard))
   error('N of clustered hypotheses do not conform with N hypotheses given by hyposCard');
end
%exiThreshold = 0.0001; % This value was too low to be used for Mechi
exiThreshold = 0.000001;

pLOld = probLogHypos;

% New version - try to only sort inside individual clusters
% NOTICE: This means that existence uncertainty also should be utilized for pruning purposes.

cBegs = tCloud2BegInd(clustersCard);
cEnds = tCloud2EndInd(clustersCard);
toBeRemoved = false(1,sum(clustersCard));

hBegs = tCloud2BegInd(hyposCard);
hEnds = tCloud2EndInd(hyposCard);

pruningPreparations = struct('toBeRemovedLocal',{},'probabilities',{},'bestUnresolved',{},'bestProbUnresolved',{});


nC = size(clustersCard,2);
for c=1:nC
   
    clusterC = clusters(1,cBegs(c):cEnds(c));  % Hypothesis indices of this cluster
    %pLC = probLogHypos(cBegs(c):cEnds(c)); % Logarithmic probabilities of these hypotheses
    
    pruningPreparations(c).bestProbUnresolved = [];
    
    if(~isempty(clusterC))
        
        % Mark for pruning based on hypothesis probabilities
        
        pruningPreparations(c).probabilities = probabilities(cBegs(c):cEnds(c));
        a=1-cumsum(pruningPreparations(c).probabilities);
        d = 1:size(a,2);
        pruningPreparations(c).toBeRemovedLocal = (a < 0.0006 & pruningPreparations(c).probabilities < 0.0006) | d > nHypoTotalMax;

        % Mark for pruning based on existence probabilities
        
        for ii=1:size(clusterC,2)
            tracksInH = hypos(1,hBegs(clusterC(ii)):hEnds(clusterC(ii)));
            exiInH = existences(tracksInH);
            pruningPreparations(c).toBeRemovedLocal(ii) = pruningPreparations(c).toBeRemovedLocal(ii) || ( ~isempty(exiInH) &&  all(exiInH < exiThreshold));
        end        
        pruningPreparations(c).bestUnresolved = find(pruningPreparations(c).toBeRemovedLocal,1,'first');
        pruningPreparations(c).bestProbUnresolved = pruningPreparations(c).probabilities(pruningPreparations(c).bestUnresolved);
                
        

        
    end
        
    if(isempty(pruningPreparations(c).bestProbUnresolved))
        
       pruningPreparations(c).toBeRemovedLocal = [];
       pruningPreparations(c).probabilities = []; 
       pruningPreparations(c).bestUnresolved = NaN; 
       pruningPreparations(c).bestProbUnresolved = -Inf; 
        
    end
    
        if(isempty(pruningPreparations(c).bestProbUnresolved))
           error('why is pruningPreparations(c).bestProbUnresolved empty?'); 
        end    
end
[toBeKept] = find(~toBeRemoved);


nHyposKeep = length(toBeKept);
% if(k==2)
% disp('after first cluster loop');
% pruningPreparations(3)
% end
% Now I have all hypotheses that I may consider removing

maxIter = nH;
iter = 0;
while(iter < maxIter)
    
    % For loop for picking values seems to be 10 times faster than struct-cell-matrix
    
    valuesToBeSorted = zeros(1,nC);
    
    
    for c=1:nC
       valuesToBeSorted(c) =  pruningPreparations(c).bestProbUnresolved;
    end
    [sortedBUP,ixBUP] = sort(valuesToBeSorted,'descend');
    

    
    % What is the last cluster whose best unresolved hypothesis has non-inf probability?
    
    lastPossibleCandidate = find(~isinf(sortedBUP),1,'last');
    ixBUP = ixBUP(1:lastPossibleCandidate);
    
    % ixBUP now contains the clusters that I'm interested in
    
    for c=1:length(ixBUP)
        
        % Identify which hypothesis I want to save from death in this iteration
        
        hToBeSavedInC = pruningPreparations(ixBUP(c)).bestUnresolved(1);
        
        % Need to keep track of how many hypotheses have been allowed in so
        % that I'm ready to terminate the process
        % This can be done by simply summing the boolean toBeKept
        
        if(~isnan(hToBeSavedInC))
            
            savedInFullOrder = clusters(bc2a(hToBeSavedInC,ixBUP(c),clustersCard));
            toBeRemoved(savedInFullOrder) = false;
            nHyposKeep = length(toBeKept);
            
            % Then I need some stuff to take this hypothesis out of the queue
            % Can I simply repeat this statement after updating toBeRemovedLocal suitably?
            
            pruningPreparations(ixBUP(c)).toBeRemovedLocal(hToBeSavedInC) = false;
            pruningPreparations(ixBUP(c)).bestUnresolved = find(pruningPreparations(ixBUP(c)).toBeRemovedLocal,1,'first');
            pruningPreparations(ixBUP(c)).bestProbUnresolved = pruningPreparations(ixBUP(c)).probabilities(pruningPreparations(ixBUP(c)).bestUnresolved);
        end
        
        if(isempty(pruningPreparations(ixBUP(c)).bestProbUnresolved))
            pruningPreparations(ixBUP(c)).toBeRemovedLocal = [];
            pruningPreparations(ixBUP(c)).probabilities = [];
            pruningPreparations(ixBUP(c)).bestUnresolved = NaN;
            pruningPreparations(ixBUP(c)).bestProbUnresolved = -Inf;
        end
        
        if(nHyposKeep > nHypoTotalMax)
            break;
        end        
    end

    
    % For every iteration determine prioritized order of clusters
    % This entails:
    % Finding the clusters that have unresolved hypotheses
    % Identifying the best unresolved hypothesis for each of these clusters
    % Sorting these clusters accordingly
    
    
    if(nHyposKeep > nHypoTotalMax)
        break;
    end
    
    
    
    iter = iter + 1;
end

[toBeKept] = find(~toBeRemoved);

% -------------------------------------------------------------------------
% Old stuff from before bid version
% -------------------------------------------------------------------------

[hyposIndPrune,hyposCardPrune] = pickIndC(sort(clusters(toBeKept)),hyposCard);

hyposPrune = hypos(:,hyposIndPrune);
probLogHyposPrune =probLogHypos(~toBeRemoved);


% I should really separate the cluster adjustment stuff in a separate function


[clustersOut,clustersCardOut] = pruningAdjustClusters2(clusters,clustersCard,toBeKept,toBeRemoved,hyposPrune,hyposCardPrune);


begsHP = tCloud2BegInd(hyposCardPrune);
endsHP = tCloud2EndInd(hyposCardPrune);

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
    
    [clustersOut,clustersCardOut] = pruningAdjustClusters2(clustersOut,clustersCardOut,toBeKept,toBeRemoved,hyposPrune,hyposCardPrune);
    hyposNew = hyposPrune;
    hyposCardNew = hyposCardPrune;
    trackFile(inCol.exi,:) = existences;
end



 