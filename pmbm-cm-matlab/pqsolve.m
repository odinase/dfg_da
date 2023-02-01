function [pqO,labelHypo] = pqsolve(pqI,bestLocal,switches,t2mForbidden,t2mEnforced,superMinVal,gainMatPostC,labelHypo,parentInTree)

% Function designed to give direct solution of assignment problem given a
% pq-entry (pqI), information in bestLocal, and one or more switched parent hypotheses 


% @pqI: Original entry from priority queue that we are to modify
% @bestLocal: Struct from parent function containing the best local hypothesis for EACH parent hypothesis.
% @switches: nCL length row vector of switch entries for relevant clusters (NaN for others)
% @t2mForbidden: assignments in pqI.meaEntireOpt that are to be disallowed
% @t2mEnforced: assignments in pqI.meaEntireOpt that are to be enforced
% @
% @



if(length(bestLocal) ~= size(switches,2))
   error('switches dont correspond to bestLocal with regard to length'); 
end


toBeSwitched = find(~isnan(switches));
nextParents = switches(toBeSwitched);

[rewardMatrixCombined,tracksAllSwitch,meaIndAllSwitch,tracksBInds,clusterPerTrack] = constructSwitchReward(pqI,toBeSwitched,nextParents,bestLocal,superMinVal,gainMatPostC);

% Solve assignment problem

% [personToItemCr,~,optRewardCr]=assign2D(-rewardMatrixCombined); % Using David Crouses JVC-based assignment solver
% optReward = - optRewardCr;
% personToItem = personToItemCr';
%meaEO = meaIndAllSwitch(personToItem);


[personToItem,optReward,upperBoundsOfScores] = crouse2D(rewardMatrixCombined);


rewsumveriA = 0;
for ii=1:length(personToItem)
   rewsumveriA = rewsumveriA + rewardMatrixCombined(ii,personToItem(ii)); 
end
optReward = rewsumveriA;
%rewsumveriA
%optReward

% Expandability test

tempRewMat = rewardMatrixCombined; % I make this one to be able to check for expandability already here.
    for ii=1:length(personToItem)
        tempRewMat(ii,personToItem(ii)) = superMinVal;
    end
expandabilityTest = any(tempRewMat > superMinVal,2);

% Construct new entry to be inserted in priority queue

[pqO,labelHypo] = pqAppendSwitch(pqI,parentInTree,bestLocal,toBeSwitched,nextParents,rewardMatrixCombined,optReward,tracksAllSwitch,tracksBInds,meaIndAllSwitch,clusterPerTrack,expandabilityTest,personToItem,labelHypo,upperBoundsOfScores);

% if(labelHypo == 3)
%    error('look at  score calculation for L3'); 
% end
