function val = phdIntegral(iC,hypos,hyposCard,clusters,clustersCard,inCol,trackFile,probLogHypos)

% Function to evaluate the total PhD integral for a given cluster
% Isnt the PHD integral simply the sum of all TTPs in the cluster?


[trackSumProbs,trackTotalProbs,includedTracks] = trackProbGivenCluster(iC,trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
val = sum(trackTotalProbs);