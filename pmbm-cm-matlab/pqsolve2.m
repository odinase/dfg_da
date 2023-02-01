function [pqO,labelHypo] = pqsolve2(pqI,bestLocal,switches,t2mForbidden,t2mEnforced,superMinVal,gainMatPostC,labelHypo,parentInTree,minVal)

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

%[rewardMatrixCombined,tracksAllSwitch,meaIndAllSwitch,tracksBInds,clusterPerTrack] = constructSwitchReward(pqI,toBeSwitched,nextParents,bestLocal,superMinVal,gainMatPostC);


if(any(~isnan(switches)) && isempty(t2mForbidden)) % Doing a switch
    toBeSwitched = find(~isnan(switches));
    nextParents = switches(toBeSwitched);
    
    
    [rewardMatrixCombined,tracksAllSwitch,meaIndAllSwitch,tracksBInds,clusterPerTrack] = constructSwitchReward(pqI,toBeSwitched,nextParents,bestLocal,superMinVal,gainMatPostC);

elseif(~isempty(t2mForbidden) && ~any(~isnan(switches))) % Doing an expansion   
    
    rewardMatrixCombined = pqI.rewardMatrix;
    rewardMatrixCombined(t2mForbidden,pqI.customer2Item(t2mForbidden)) = superMinVal;
    tracksAllSwitch = pqI.tracksEntire;
    meaIndAllSwitch = pqI.meaEntire;
    tracksBInds = pqI.tracksBLevel;
    clusterPerTrack = pqI.tracksCLevel;
    
    
else
    error('I can only switch or expand separately so far');
    
end


% Solve assignment problem

% [personToItemCr,~,optRewardCr]=assign2D(-rewardMatrixCombined); % Using David Crouses JVC-based assignment solver
% optReward = - optRewardCr;
% personToItem = personToItemCr';
%meaEO = meaIndAllSwitch(personToItem);


[personToItem,optReward,upperBoundsOfScores] = crouse2D(rewardMatrixCombined);

% tracksAllSwitch
% meaIndAllSwitch
% rewardMatrixCombined
% personToItem
% optReward

if(optReward >= minVal)
    
    % Expandability test
    
    tempRewMat = rewardMatrixCombined; % I make this one to be able to check for expandability already here.
    %tempRewMat(:,personToItem) = superMinVal;
    for ii=1:length(personToItem)
        tempRewMat(ii,personToItem(ii)) = superMinVal;
    end
    
    
    expandabilityTest = any(tempRewMat > superMinVal,2);
    
    % Construct new entry to be inserted in priority queue
    
    [pqO,labelHypo] = pqAppend(pqI,parentInTree,bestLocal,toBeSwitched,nextParents,t2mForbidden,rewardMatrixCombined,optReward,tracksAllSwitch,tracksBInds,meaIndAllSwitch,clusterPerTrack,expandabilityTest,personToItem,labelHypo,upperBoundsOfScores);
    
%     if(pqO.labelHypo == 7)
%         error('Look at  HL7');
%     end
    
else
   pqO = zeros(1,0); % So that I can test for whether pqO is empty 
    
end