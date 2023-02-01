function [ttp,tracksCol,hyposCol,clustersCol,clustersAndTTP] = ttpSegment(seg,k1,k2,kEval,meaHistCol,trackFile,hypos,hyposCard,clusters,clustersCard,probLogHypos,inCol)

% Function to evaluate the total track probability of all tracks at time
% kEval that conforms with track segment seg between k1 and k2.
% Written by Edmund Brekke during March 2021
% @seg:             Segment of measurent indices, i.e. part of track
% @k1:              First time step in seg
% @k2:              Last time step in seg
% @kEval:           Last time step in meaHistCol
% @meaHistCol:      Matrix of all tracks as measurement indices
% @trackFile:       Track file consisting kinematic info, existence probabilities etc.
% @hypos:           Hypothesis array containing track indices
% @hyposCard:       Corresponding cardinality list
% @clusters:        Cluster array containing hypothesis indices
% @custersCard:     Corresponding cardinality list
% @probLogHypos:    Logarithmic probabilities of the hypotheses
% @inCol:           Meanings of rows in trackFile
% >ttp              Sum of total track probabilities for all tracks that conform with seg
% >tracksCol        All tracks that conform with seg
% >hyposCol         All hypotheses that conform with seg - cell array
% >clustersCol      The corresponding clusters

% Be aware: The evaluated TTP can exceed one if the segment is found in multiple clusters.


if(k2+1-k1 ~= size(seg,1))
   error('Time steps do not match length of seg'); 
end

mhc = kEval-size(meaHistCol,1)+1:kEval; % Time steps of rows in meaHistCol

msBeg = max(k1,mhc(1));
msEnd = mhc(end);

msBegInSeg = msBeg - k1 + 1;
meaSeqFirstPart = seg(msBegInSeg:end);
remainingSteps = kEval - k2;
meaSeqLastPart = NaN*ones(remainingSteps,1);
meaSeq = [meaSeqFirstPart; meaSeqLastPart];

[tracks,hyposCol,clusterNumbers] = findTrackAll(meaSeq,meaHistCol,hypos,hyposCard,clusters,clustersCard);
tracksCol = tracks;
[~,trackTotalProbs] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
ttp = sum(trackTotalProbs(tracks));
clustersCol = clusterNumbers;

%mh = meaHistCol(:,tracks)
%meaSeq


% Shall I also or instead evaluate TTP per cluster?
% 

[u,~,vb] = unique(clustersCol);

% Remove non-valid clusters. It works because Matlab places NaNs last. 

ix = ~isnan(u);
u = u(:,ix);

% Assemble the valid clusters

nClusters = length(u);
clustersAndTTP = zeros(2,nClusters);
clustersAndTTP(1,:) = u;

% Calculate TTP for each cluster

for ii=1:length(u)
   clustersAndTTP(2,ii) = sum(trackTotalProbs(vb == ii));
end










