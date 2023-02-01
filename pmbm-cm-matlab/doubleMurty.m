function [hyposLocal,hyposCardLocal,probLogLocal] = doubleMurty(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoTotalMax,trackNumberLookup,k)

% Double Murty to be used for hypothesis generation in each supercluster in PMBM.
% Written by Edmund Brekke during 2020-2021
% Based on original Murty implementation by Umut Orguner from tracking
% course at Linkoping University.
% @hypos            Track numbers group together in the hypotheses
% @hyposCard        Cardinality array of hypos
% @clusters         Hypothesis numbers grouped together in clusters
% @clustersCard     Cardinality array of clusters
% @probLogHypos     Logaritm of probabilities for all hypotheses
% @iC               Number of super cluster (i.e., master cluster)
% @assocLocal       2-row association array for clusters
% @gainMatPostC:    Complete reward matrix. Together with next argument it gives me the integer nT.
% @indicesOfNewbornTracks: Newborn track index for each measurement. Gives me the integer m.
% @nHypoMax:        Max # hypotheses in the inner Murty calls.
% @nHypoTotalMax:   Total maximum of allowable hypotheses per cluster
% @trackNumberLookup: Every finite entry in gainMatPostC corresponds to a track number in posterior track collection. 
% >hyposLocal       Track numbers group together in the new hypotheses for super cluster iC
% >hyposLocalCard   Corresponding hypothesis cardinalities.
% >probLogLocal     Corresponding logarithmic hypothesis probabilities.
% $pickIndC
% $MurtyBB
% $auctionExtended


m = size(indicesOfNewbornTracks,2);
nT = size(gainMatPostC,1)-m;
lMatPostC = gainMatPostC(1:nT,1:m);

masters = find(assocLocal(2,:) == 1);

bestScoreInC = -Inf; % New dynamic parameter to be used for branch and bound purposes

intoThis = find(assocLocal(1,:) == masters(iC)); % Find all clusters that need to be merged with original master iC
contentThis = clusters(pickIndC(intoThis,clustersCard)); % This would be the hypotheses of to-be-clustered clusters.

% Construct reward matrix to be used by clustering-purpose Murty

firstContent = clusters(pickIndC(intoThis(1),clustersCard));
firstScores = probLogHypos(firstContent);
rewardFC = firstScores;
for jC=2:length(intoThis)
    nextContent = clusters(pickIndC(intoThis(jC),clustersCard));
    nextScores = probLogHypos(nextContent);
    n1 = size(rewardFC,1);
    n2 = size(rewardFC,2);
    rewardFC = blkdiag(rewardFC,nextScores);
    rewardFC((n1+1):end,1:n2) = -Inf;
    rewardFC(1:n1,(n2+1):end) = -Inf;
end
%nHMaxInClustering = 30;
nHMaxInClustering = max(100,max(clustersCard(intoThis)));

% Run Murty for clustering

minval=min(rewardFC(~isinf(rewardFC)))-6.9;
logThresholdPrune = 6.9;
DynamicThreshold = minval;

% -------------------------------------------------------------------------
% Do the first iteration of outer Murty before entering its while loop
% -------------------------------------------------------------------------

RewardMatrix = rewardFC;
N = nHMaxInClustering;

[NofCustomers,NofItems]=size(RewardMatrix);%Get the number of customers and items
Customer2Item=zeros(N,NofCustomers);%Define the output array for customer assignments
Item2Customer=zeros(N,NofItems);%Define the output array for item assignments (Assignment 0 means no assignment)
Rewards=zeros(N,1);%Define the reward values for the solutions
[FirstCustomer2Item,FirstItem2Customer]=auctionExtended(RewardMatrix); %Get the best solutions

DynamicRewardMatrix{1}=RewardMatrix;%Assignment matrix of the Problems in the List
DynamicCustomer2Item=FirstCustomer2Item;%Customer Assignments for the Solutions in the List
DynamicItem2Customer=FirstItem2Customer;%Item Assignments for the Solutions in the List
DynamicReward=RewardCalculate(RewardMatrix,FirstCustomer2Item);%Rewards for the Solutions in the List


% Already here I should explore hypothesis expansion for the first clustered combination

rowOfHypos = contentThis(FirstCustomer2Item(1,:));
parentHypo = hypos(pickIndC(rowOfHypos,hyposCard));  % What I add here is track labels
parentProbLog = DynamicReward;

hInC = contentThis;
tInC = hypos(pickIndC(hInC,hyposCard));
gatedInC = find(any(~isinf(lMatPostC(tInC,:)),1));
gatedInCAsNewBornTracks = indicesOfNewbornTracks(gatedInC);

% Preallocate memory to hyposLocal etc.

nTL = 0;
for ii=1:size(intoThis,2)
    contentI = clusters(pickIndC(intoThis(ii),clustersCard));
    nTL = nTL + max(hyposCard(contentI));
end
nML = size(gatedInC,2);

% Maximal sizes of local hypotheses and cardinality arrays

hLAlloc = (nTL+nML)*(N+1)*nHypoMax;
cLAlloc = (N+1)*nHypoMax;
pointerHL = 0;
pointerCL = 0;

hyposLocal = NaN*zeros(1,hLAlloc);
hyposCardLocal = NaN*zeros(1,cLAlloc);
probLogLocal = NaN*zeros(1,cLAlloc);



if(~isempty(parentHypo))
    if(isinf(gainMatPostC(parentHypo(1),parentHypo(1)+m)))
        error('Misdetection entry in gainMatPostC is Inf');
    end
    
    downIndices = parentHypo;  % The real and virtual track numbers (prior numbering) that are to be considered under parentHypo
    alongIndices = [gatedInC,m+parentHypo];
    gainMatH = gainMatPostC(downIndices,alongIndices);
    
    % murtyRevised should be replaced by MurtyBB
    
    [C2I,I2C,NSol,RWS,DynamicThreshold]=MurtyBB(gainMatH,nHypoMax,DynamicThreshold,logThresholdPrune,parentProbLog);
    
    for hM=1:NSol
        meaByMurty = alongIndices(C2I(hM,:));  % Measurements assigned to existing tracks
        
        % How to extract measurements that are validated but which under current hypothesis constitute newborn tracks?
        
        hNewLegacy = zeros(1,size(parentHypo,2));
        for ii=1:size(hNewLegacy,2)
            hNewLegacy(1,ii) = trackNumberLookup(downIndices(ii),meaByMurty(ii));
        end
        newBornTracksToActivate = indicesOfNewbornTracks(setdiff(gatedInC,meaByMurty));
        hNew = [hNewLegacy,newBornTracksToActivate];
        
        hyposLocal(:,(pointerHL+1):(pointerHL+size(hNew,2))) = hNew;
        hyposCardLocal(:,pointerCL+1) = size(hNew,2);
        probLogLocal(:,pointerCL+1) = RWS(hM);
        pointerHL = pointerHL + size(hNew,2);
        pointerCL = pointerCL + 1;
    end
else
    hNew = indicesOfNewbornTracks(gatedInC);
    newMeaPostLogs = diag(gainMatPostC(nT:end,1:m));
    hyposLocal(:,(pointerHL+1):(pointerHL+size(hNew,2))) = hNew;
    hyposCardLocal(:,pointerCL+1) = size(hNew,2);
    probLogLocal(:,pointerCL+1) = parentProbLog + sum(newMeaPostLogs(gatedInC));
    pointerHL = pointerHL + size(hNew,2);
    pointerCL = pointerCL + 1;
    
end

% -------------------------------------------------------------------------
% The outer Murty while loop - continues the Murty process already started
% -------------------------------------------------------------------------

RewardMatrix(isinf(RewardMatrix)) = minval;

iM=0;%Set the index to zero.
while  (iM<=N && ~isempty(DynamicReward))
    
    
    iM=iM+1; %increment i.
    
    [maxval indmax]=max(DynamicReward);%Find the maximum reward solution in the list
    TempCustomer2Item=DynamicCustomer2Item(indmax,:);%get the customer assignments of the solution
    TempItem2Customer=DynamicItem2Customer(indmax,:);%get the item assignments of the solution
    TempRewardMatrix=DynamicRewardMatrix{indmax};%get the rewardmatrix of the problem
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Add the maximum reward solution in the list to the list of solutions
    %to be returned
    Customer2Item(iM,:)=TempCustomer2Item;%Set the ith best solution's customer assignments
    Item2Customer(iM,:)=TempItem2Customer;%Set the ith best solution's item assignments
    Rewards(iM)=maxval;%Set the ith best solution's reward
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Remove the maximum reward solution from the list
    DynamicReward(indmax)=[];%Remove the reward from the list
    DynamicCustomer2Item(indmax,:)=[];%Remove the customer assignments from the list
    DynamicItem2Customer(indmax,:)=[];%Remove the item assignments from thel list
    DynamicRewardMatrix(indmax)=[];%Remove the assignment matrix of the problem from the list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (iM==N)%if we have already determined the Nth best solution
        break;%no need to continue to spend computation time, get out of the loop.
    end
    
    for j=1:NofCustomers %Trace all associations of the given solution
        TempRewardMatrix(j,TempCustomer2Item(j))=minval;%Make the customer association invalid
        if(any(all(isinf(TempRewardMatrix),2),1))
            ValidInvalidFlag = false;    % Do nothing in this case
        else
            % Valid solution exists -> Add solution, reward and reward matrix to list
            
            [LoopCustomer2Item,LoopItem2Customer]=auctionExtended(TempRewardMatrix);%solve the modified problem
            Rew=RewardCalculate(TempRewardMatrix,LoopCustomer2Item);
            if(Rew > DynamicThreshold)
                ValidInvalidFlag = true;
                
                % Adjust dynamic threshold to avoid exploring unreasonable hypotheses
                
                DynamicThreshold = max(DynamicThreshold,Rew-logThresholdPrune);
            else
                ValidInvalidFlag = false;
            end
        end
        
        %ValidInvalidFlag=CheckValidity(TempRewardMatrix,LoopCustomer2Item,minval);%check the validity of the new solution
        
        if ValidInvalidFlag %if the found solution is valid
            Ndynamic=length(DynamicReward);%find the number of problem-solution pains in the list
            DynamicReward(Ndynamic+1)=RewardCalculate(TempRewardMatrix,LoopCustomer2Item);%Add the reward of thesolution to the list
            DynamicCustomer2Item(Ndynamic+1,:)=LoopCustomer2Item;%Add customer assignments to the list
            DynamicItem2Customer(Ndynamic+1,:)=LoopItem2Customer;%Add the item assignments to the list
            DynamicRewardMatrix{Ndynamic+1}=TempRewardMatrix;%Add the assignment matrix of the problem to the list
            
            % ----------------------------------------------
            % Within this if-clause hypothesis generation should find place
            % ----------------------------------------------
            
            rowOfHypos = contentThis(LoopCustomer2Item(1,:));
            parentHypo = hypos(pickIndC(rowOfHypos,hyposCard));
            parentProbLog = DynamicReward(Ndynamic+1);
            
            if(~isempty(parentHypo))
                
                downIndices = parentHypo;  % The real and virtual track numbers (prior numbering) that are to be considered under parentHypo
                hInC = contentThis;
                tInC = hypos(pickIndC(hInC,hyposCard));
                gatedInC = find(any(~isinf(lMatPostC(tInC,:)),1));
                gatedInCAsNewBornTracks = indicesOfNewbornTracks(gatedInC);
                alongIndices = [gatedInC,m+parentHypo];
                gainMatH = gainMatPostC(downIndices,alongIndices);
                
                % Call to inner Murty loop in separate function
                
                %nHypoMaxDynamic = min(nHypoMax,max(0,nHypoTotalMax - pointerCL));
                nHypoMaxDynamic = nHypoMax;

                
                
                if(nHypoMaxDynamic > 0)
                    [C2I,I2C,NSol,RWS,DynamicThreshold]=MurtyBB(gainMatH,nHypoMaxDynamic,DynamicThreshold,logThresholdPrune,parentProbLog);
                    
                    for hM=1:NSol
                        meaByMurty = alongIndices(C2I(hM,:));  % Measurements assigned to existing tracks
                        
                        % How to extract measurements that are validated but which under current hypothesis constitute newborn tracks?
                        
                        hNewLegacy = zeros(1,size(parentHypo,2));
                        for ii=1:size(hNewLegacy,2)
                            hNewLegacy(1,ii) = trackNumberLookup(downIndices(ii),meaByMurty(ii));
                        end
                        newBornTracksToActivate = indicesOfNewbornTracks(setdiff(gatedInC,meaByMurty));
                        hNew = [hNewLegacy,newBornTracksToActivate];
                        %probLogLocal = [probLogLocal,parentProbLog+RWS(hM)];
                        hyposLocal(:,(pointerHL+1):(pointerHL+size(hNew,2))) = hNew;
                        hyposCardLocal(:,pointerCL+1) = size(hNew,2);
                        probLogLocal(:,pointerCL+1) = RWS(hM);
                        pointerHL = pointerHL + size(hNew,2);
                        pointerCL = pointerCL + 1;
                        
                    end
                end
            else
                
                nHypoMaxDynamic = min(nHypoMax,max(0,nHypoTotalMax - pointerCL));
                if(nHypoMaxDynamic > 0)
                    newMeaPostLogs = diag(gainMatPostC(nT:end,1:m));
                    hNew = indicesOfNewbornTracks(gatedInC);
                    hyposLocal(:,(pointerHL+1):(pointerHL+size(hNew,2))) = hNew;
                    hyposCardLocal(:,pointerCL+1) = size(hNew,2);
                    probLogLocal(:,pointerCL+1) = parentProbLog + sum(newMeaPostLogs(gatedInC));
                    pointerHL = pointerHL + size(hNew,2);
                    pointerCL = pointerCL + 1;
                end
            end
        end
        %restore the reward value
        TempRewardMatrix(j,TempCustomer2Item(j))=RewardMatrix(j,TempCustomer2Item(j));%Make the customer association valid again
        %remove all possible associations except the selected one
        %i.e., make all association except the customer association invalid
        TempRewardMatrix(j,1:(TempCustomer2Item(j)-1))=minval;
        TempRewardMatrix(j,(TempCustomer2Item(j)+1):end)=minval;
        TempRewardMatrix(1:(j-1),TempCustomer2Item(j))=minval;
        TempRewardMatrix((j+1):end,TempCustomer2Item(j))=minval;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
end





% -------------------------------------------------------------------------
% Done with Double Murty: Prepare output arrays for returning
% -------------------------------------------------------------------------

hyposLocal(:,pointerHL+1:end) = [];
hyposCardLocal(:,pointerCL+1:end) = [];
probLogLocal(:,pointerCL+1:end) = [];
