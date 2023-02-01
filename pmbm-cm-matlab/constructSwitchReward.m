function [rewardMatrixCombined,tracksAllSwitch,meaIndAllSwitch,tracksBInds,clusterPerTrack] = constructSwitchReward(pqI,toBeSwitched,nextParents,bestLocal,superMinVal,gainMatPostC)

% Function to construct reward matrix etc. used in switches
% Written by Edmund Brekke during October 2021
% @pqI: Parent entry from priority queue of global hypotheses
% @toBeSwitched: Contains numbers of clusters where I plan to switch parent hypotheses.
% @nextParents: The parents that I shall switch to. 
% @bestLocal: Struct from parent function containing the best local hypothesis for EACH parent hypothesis.
% @iCluster
%
%
%


% Rationale: Involves several messy index operations that I would like to
% separate from other code.


nCL = length(bestLocal);

tracksISwitch = zeros(1,0);
preSortClustersTracks = zeros(1,0);
for jj=1:length(toBeSwitched)
    tracksISwitch = [tracksISwitch,bestLocal(toBeSwitched(jj)).clustercontrib(nextParents(jj)).tracks];
    preSortClustersTracks = [preSortClustersTracks,toBeSwitched(jj)*ones(size(bestLocal(toBeSwitched(jj)).clustercontrib(nextParents(jj)).tracks))];
end
%= bestLocal(ii).clustercontrib(nextParentI).tracks;

tracksLegacySwitch = zeros(1,0);
tracksLegacySwitchLocalInSuper = zeros(1,0);
%preSortClustersTracks = iCluster*ones(size(tracksISwitch));
tracksBInds = 1:size(preSortClustersTracks,2);

for jj=setdiff(1:nCL,toBeSwitched)
    tracksLegacySwitch = [tracksLegacySwitch, pqI.clustercontrib(jj).tracks];
    tracksLegacySwitchLocalInSuper = [tracksLegacySwitchLocalInSuper, pqI.clustercontrib(jj).tracksLocalInSuper];
    preSortClustersTracks = [preSortClustersTracks,jj*ones(size(pqI.clustercontrib(jj).tracks))];
    tracksBInds = [tracksBInds,1:size(pqI.clustercontrib(jj).tracks,2)];
end

meaIndLegacySwitchInSuper = find(any(pqI.rewardMatrix(tracksLegacySwitchLocalInSuper,:) > superMinVal,1));
if(isempty(meaIndLegacySwitchInSuper))
    meaIndLegacySwitchInSuper = zeros(1,0);
end
meaIndLegacySwitch = pqI.meaEntire(meaIndLegacySwitchInSuper);
meaIndISwitch = find(any(gainMatPostC(tracksISwitch,:) > superMinVal,1));
meaIndAllSwitch = union(meaIndISwitch,meaIndLegacySwitch);
meaIndINotLegacy = setdiff(meaIndISwitch,meaIndLegacySwitch); % I checked that this makes sense :)

% Construct the switched and legacy parts of the new reward matrix.

rewardMatrixISwitch = gainMatPostC(tracksISwitch,meaIndAllSwitch);
rewardMatrixLegacyRaw = pqI.rewardMatrix(tracksLegacySwitchLocalInSuper,meaIndLegacySwitchInSuper);

[clusterPerTrack,ix] = sort(preSortClustersTracks,'ascend');
tracksAllSwitchPreSort = [tracksISwitch,tracksLegacySwitch];
tracksAllSwitch = tracksAllSwitchPreSort(ix);

[a,b]=ismember(meaIndLegacySwitch,meaIndAllSwitch); %n gives me where to put the active columsn of rewardMatrixLegacyRaw
rewardMatrixLegacyExpanded = superMinVal*ones(size(rewardMatrixLegacyRaw,1),size(rewardMatrixLegacyRaw,2)+size(meaIndINotLegacy,2));
rewardMatrixLegacyExpanded(:,b) = rewardMatrixLegacyRaw;
rewardMatrixCombinedRaw = [rewardMatrixISwitch; rewardMatrixLegacyExpanded];
rewardMatrixCombined = rewardMatrixCombinedRaw(ix,:);  % And now reorder so that I get the clusters in the right order
tracksBInds = tracksBInds(ix);


bool = zeros(1,max(clusterPerTrack));
for ii=1:max(clusterPerTrack)
    bEntries = tracksBInds(clusterPerTrack == ii);
    if(~isempty(bEntries))
        bool(ii) = max(bEntries) == length(bEntries);
    else
        bool(ii) = true; 
    end
end
if(any(~bool))
   error('something goes wrong with tracksBInds'); 
end

% if(nextParents == 2 && pqI.labelHypo == 14)
%    error('what happens here'); 
% end


