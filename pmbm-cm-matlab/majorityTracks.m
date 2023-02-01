function [selectedTracks,trackSumProbs] = majorityTracks(hypos,hyposCard,clusters,clustersCard,probLogHypos,sig,trackFile,inCol)

% Function to identify all tracks that claim more than 1-sig of the
% probability mass in their clusters

% @hypos        Track numbers group together in the hypotheses
% @hyposCard    Cardinality array of hypos
% @clusters     Hypothesis numbers grouped together in clusters
% @clustersCard Cardinality array of clusters
% @probLogHypos     Logaritm of probabilities for all hypotheses
% @sig          Significance level for exclusion from majority


% Do I have a version of trackProbAccumulatePure that only returns trackSumProbs?
% Then it wouldnt have to be passed trackFile as an argument
% Or just copy paste its content into this function?



[trackSumProbs] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
selectedTracks = trackSumProbs > 1-sig;