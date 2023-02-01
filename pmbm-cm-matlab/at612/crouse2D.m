function [personToItem,optReward,upperBoundsOfScores] = crouse2D(rewardMat)

% This function is simply a wrapper of David Crouse's assign2D.m
% Its purpose is to enable a systematic utilization of the dual variables 
% u and v for devising upper bounds for rewards.
% Written by Edmund Brekke during February 2022

costMat = -rewardMat;
minCost = min(costMat(:));
pMat = costMat - minCost;
nCustomers = size(rewardMat,1);
nItems = size(rewardMat,2);
[personToItemCr,~,optCostCr,u,v]=assign2D(pMat);
boundsMat = optCostCr + pMat - repmat(v,[1,nItems]) - repmat(u',[nCustomers,1]);
boundsMat2 = boundsMat + minCost*nCustomers;
optCostCr2 = optCostCr + minCost*nCustomers;
optReward = - optCostCr2;
upperBoundsOfScores = -boundsMat2;
personToItem = personToItemCr';