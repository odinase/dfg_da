function [ass,valSum] = nonAuction(RewardMatrix)

% Function to generate cardinality combinations in a Murty-Auction fashion without the standard one-item-per-customer constraint
% Written by Edmund Brekke during July 2020

[maxVals,ass] = max(RewardMatrix,[],2);
valSum = sum(maxVals);
ass = ass';
