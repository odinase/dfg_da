function [personToItem,optReward,upperBoundsOfScores] = crouseWrap(rewardMat)

costMat = -gainMat;
minCost = min(costMat(:));
pMat = costMat - minCost;
nCustomers = size(gainMat,1);
nItems = size(gainMat,2);
[personToItemCr,~,optCostCr,u,v]=assign2D(pMat);



