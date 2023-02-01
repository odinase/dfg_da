function pq = pqInitialize(bestLocal,labelHypo,gainMatPostC,superMinVal)

% Function to construct initial hypothesis in B-and-B search tree
% Written by Edmund Brekke during October 2021
% @bestLocal: Struct from parent function containing the best local hypothesis for EACH parent hypothesis.
% @labelHypo: Counter for generating hypothesis labels
% @gainMatPostC: Complete reward matrix after pre-cluster preparations
% @superMinVal: Dummy variable for impossible rewards
% >pq: Struct used to represent the priority queue of global hypotheses
% $expandabilityTest
% $assign2D: David Crouses assignment solver

% Key problem solved by this function:
% The best local hypotheses from different clusters may be incompatible.
% Therefore a global assignment problem must be solved.

% Overall pq fields:
% score
% rewardMatrix
% tracksEntire
% meaEntire
% meaEntireOpt
% Clustercontrib fields:
% tracks
% meaIndAll
% meaIndOpt
% meaIndOptLocal
% parentScore

% Explanation of fields of pq:
% tracksBLevel: For each row in rewardMatrix: Where is the track in the local list of its cluster (ABC-bookkeeping).
% tracksCLevel: For each row in rewardMatrix: Which cluster does the track belong to  (ABC-bookkeeping).
% customer2Item: Which column is assigned to each row in the solution of rewardMatrix.


nCL = length(bestLocal);


pq = struct('score',{},'bound',{},'rewardMatrix',{},'tracksEntire',{},'meaEntire',{},'meaEntireOpt',{},'clustercontrib',{},...
    'tracksBLevel',{},'tracksCLevel',{},'customer2Item',{},'parentInSearchTree',{},'childrenInSearchTree',{},'childrenCount',{},...
    'labelHypo',{},'switchOrExpand',{},'bestCaseLoss',{},'upperBoundsOfScores',{},'hasBeenUUB',{});

% Assemble best combo of parent hypotheses.

tracks = zeros(2,0); % 1st row parent tracks, 2nd row contributing cluster
meaIndices = zeros(2,0); % 1st row measurement index, 2nd row contributing cluster

uubParents = ones(1,nCL); % The parents of holder of UUB, which by definition will be this hypothesis

tracksBLevel = zeros(1,0);
for ii=1:nCL
    tracks = [tracks,[bestLocal(ii).clustercontrib(1).tracks; ii*ones(size(bestLocal(ii).clustercontrib(1).tracks))]];
    tracksBLevel = [tracksBLevel,1:size(bestLocal(ii).clustercontrib(1).tracks,2)];
    meaIndices = [meaIndices, [bestLocal(ii).clustercontrib(1).meaIndAll; ii*ones(size(bestLocal(ii).clustercontrib(1).meaIndAll))]];
end

[meaIndUnique,~,cMea] = unique(meaIndices(1,:));

% Construct and solve assignment problem for the first combo

if(~isempty(tracks))
    
    rewardMatrix = gainMatPostC(tracks(1,:),meaIndUnique);
    %[personToItemCr,~,optRewardCr]=assign2D(-rewardMatrix);
    [personToItem,optReward,upperBoundsOfScores] = crouse2D(rewardMatrix);
    %optReward = - optRewardCr;
    %personToItem = personToItemCr';
    
    
    if(optReward < -500)
       error('too low optreward'); 
    end
    
    % Construct the first element of the priority queue
    
    pq(1).bound = 0;
    nHL = zeros(1,nCL);
    for ii=1:nCL
        pq(1).bound = pq(1).bound + bestLocal(ii).scores(1);
        nHL(ii) = length(bestLocal(ii).scores);
    end
    
    pq(1).score = optReward;
    pq(1).bestCaseLoss = -Inf*ones(nCL,max(nHL));
    for ii=1:nCL
        pq(1).score = pq(1).score + bestLocal(ii).clustercontrib(1).parentPriorScore;
    end
    pq(1).rewardMatrix = rewardMatrix;
    pq(1).tracksEntire = tracks(1,:);
    pq(1).meaEntire = meaIndUnique;
    pq(1).meaEntireOpt = meaIndUnique(personToItem);
    pq(1).tracksCLevel = tracks(2,:);
    pq(1).tracksBLevel = tracksBLevel;
    pq(1).customer2Item = personToItem;
    pq(1).parentInSearchTree = 0; % This is the root node.
    pq(1).labelHypo = labelHypo; % This is the root node.
    pq(1).childrenInSearchTree = zeros(1,0); % This is the root node.
    pq(1).childrenCount = 0;
    
    pq(1).hasBeenUUB = false(1,1);
    
    pq(1).switchOrExpand = struct('switch',false,'expand',false,'clusterBase',[],'trackBase',[]);
    
    pq(1).clustercontrib = struct('tracks',{},'parentInBestList',{},'parentPriorScore',{},'expandable',{},'switchable',{},...
        'tracksLocalInSuper',{},'meaIndOptLocalInSuper',{},'posteriorScore',{}); % Let me just start with this and then see what more I need.
    
    tempRewMat = rewardMatrix; % I make this one to be able to check for expandability already here.
    for ii=1:length(personToItem)
        tempRewMat(ii,personToItem(ii)) = superMinVal;
    end
    %tempRewMat(:,personToItem) = superMinVal;
    expandabilityTest = any(tempRewMat > superMinVal,2); % Expandability depends on tracks having alternative measurements
    
     %pq(1).upperBoundsOfScores = superMinVal*ones(size(rewardMatrix));
     pq(1).upperBoundsOfScores = upperBoundsOfScores;   
     
    for ii=1:nCL
        pq(1).clustercontrib(ii).parentInBestList = 1;
        pq(1).clustercontrib(ii).tracks = bestLocal(ii).clustercontrib(1).tracks;
        pq(1).clustercontrib(ii).parentPriorScore = bestLocal(ii).clustercontrib(1).parentPriorScore;
        [~,b] = ismember(pq(1).clustercontrib(ii).tracks,pq(1).tracksEntire);
        pq(1).clustercontrib(ii).tracksLocalInSuper = b;
        
        % Check for expandability already here to rule out impossible branches
        
        pq(1).clustercontrib(ii).expandable = expandabilityTest(pq(1).clustercontrib(ii).tracksLocalInSuper)';
        
        % Switchability depends on whether alternative parents in the same cluster exist
        
        pq(1).clustercontrib(ii).switchable = length(bestLocal(ii).clustercontrib) > 1;
        if(~isempty(pq(1).clustercontrib(ii).tracks))
            [~,b] = ismember(pq(1).meaEntireOpt(pq(1).tracksCLevel==ii),pq(1).meaEntire);
            pq(1).clustercontrib(ii).meaIndOptLocalInSuper = b;
            rewardSumsI = sum(pq(1).rewardMatrix(m2v([pq(1).clustercontrib(ii).tracksLocalInSuper;pq(1).clustercontrib(ii).meaIndOptLocalInSuper],size(pq(1).rewardMatrix))));
            contribScoreI = rewardSumsI + bestLocal(ii).clustercontrib(1).parentPriorScore;
            pq(1).clustercontrib(ii).posteriorScore = contribScoreI;
            
            [~,meaIndLocalInSuper] = ismember(bestLocal(ii).clustercontrib(1).meaIndAll,meaIndUnique);
            %pq(1).upperBoundsOfScores(pq(1).clustercontrib(ii).tracksLocalInSuper,meaIndLocalInSuper) = bestLocal(ii).clustercontrib(1).upperBoundsOfScores;
            
        else
            pq(1).clustercontrib(ii).posteriorScore = bestLocal(ii).clustercontrib(1).parentPriorScore; 
        end
        for hh=1:(nHL(ii))
            
            
            % SHOULD I REVISE THIS ONE TO CONTAIN ACTUAL SCORES?
            %pq(1).bestCaseLoss(ii,hh) = bestLocal(ii).scores(hh)-bestLocal(ii).scores(1);
            pq(1).bestCaseLoss(ii,hh) = bestLocal(ii).scores(hh);
        end
        
        
        
        
        
        
    end
    
   

else
    
    pq(1).score = 0;
    for ii=1:nCL
        pq(1).score = pq(1).score + bestLocal(ii).clustercontrib(1).parentPriorScore;
    end
    pq(1).rewardMatrix = [];
    pq(1).tracksEntire = zeros(1,0);
    pq(1).meaEntire = zeros(1,0);
    pq(1).meaEntireOpt = zeros(1,0);
    pq(1).tracksCLevel = zeros(1,0);
    pq(1).tracksBLevel = zeros(1,0);
    pq(1).customer2Item = zeros(1,0);
    pq(1).parentInSearchTree = 0; % This is the root node.
    pq(1).labelHypo = labelHypo; % This is the root node.
    
    pq(1).switchOrExpand = struct('switch',false,'expand',false,'clusterBase',[],'trackBase',[]);
    
    pq(1).clustercontrib = struct('tracks',{},'parentInBestList',{},'parentPriorScore',{},'expandable',{},'switchable',{},...
        'tracksLocalInSuper',{},'meaIndOptLocalInSuper',{},'posteriorScore',{}); % Let me just start with this and then see what more I need.
    
    pq(1).hasBeenUUB = false(1,1);
    
    for ii=1:nCL
        pq(1).clustercontrib(ii).parentInBestList = 1;
        pq(1).clustercontrib(ii).tracks = zeros(1,0);
        pq(1).clustercontrib(ii).parentPriorScore = bestLocal(ii).clustercontrib(1).parentPriorScore;
        pq(1).clustercontrib(ii).tracksLocalInSuper = zeros(1,0);
        
        % Check for expandability already here to rule out impossible branches
        
        pq(1).clustercontrib(ii).expandable = false(1,0);
        
        % Switchability depends on whether alternative parents in the same cluster exist
        
        pq(1).clustercontrib(ii).switchable = length(bestLocal(ii).clustercontrib) > 1;
        pq(1).clustercontrib(ii).meaIndOptLocalInSuper = zeros(1,0);
        
        pq(1).clustercontrib(ii).posteriorScore = bestLocal(ii).clustercontrib(1).parentPriorScore;
        
        
        if(isempty(pq(1).clustercontrib(ii).posteriorScore))
            error('Why empty posterior score');
        end
        
    end
    pq(1).bound = 0;
    
    nHL = zeros(1,nCL);
    for ii=1:nCL
        pq(1).bound = pq(1).bound + bestLocal(ii).scores(1);
        nHL(ii) = length(bestLocal(ii).scores);
    end
    
    pq(1).bestCaseLoss = -Inf*ones(nCL,max(nHL));
    for ii=1:nCL
        for hh=1:(nHL(ii))
            pq(1).bestCaseLoss(ii,hh) = bestLocal(ii).scores(hh);
        end
    end
     pq(1).upperBoundsOfScores = pq(1).rewardMatrix;
end
% gapReductions = gapReduce(pq(1),bestLocal);
% pq(1).bound = pq(1).bound - sum(gapReductions);


%error('kkk');
