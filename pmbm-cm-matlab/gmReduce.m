function [etaAve,covAve] = gmReduce(etaList,covList,weights)

% @etaList: dim x n array where n is number of estimates
% @covList: dim x dim x (n+1) array where n is number of estimates.
%           Last matrix is zeroth covariance
% @weights: 1 x (n+1) array where last entry is zeroth weight (betaZero)

n = size(etaList,2);
dimX = size(etaList,1);
weights = weights/sum(weights);
activeWeight = sum(weights(1:n));
etaAve = etaList*weights(1:n)'/activeWeight;
covAve = zeros(dimX,dimX);
for ii=1:n
    nuI = etaList(:,ii) - etaAve;
    covAve = covAve + weights(ii)*(nuI*nuI') + weights(ii)*covList(:,:,ii);
end




