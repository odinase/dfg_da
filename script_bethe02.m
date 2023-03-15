clear all;

addpath('./pmbm-cm-matlab');
addpath('./pmbm-cm-matlab/PMBM filter');
addpath('./pmbm-cm-matlab/GOSPA code');
addpath('./pmbm-cm-matlab/Assignment');
addpath('./pmbm-cm-matlab/at612');
addpath('./pmbm-cm-matlab/at612/jmpdFunctions');
addpath('./lightspeed');

rewMat = -Inf*ones(5,7);
rewMat(:,1) = [3.0 ; 3.2; -3; -Inf; -Inf];
rewMat(:,2) = [-Inf; -Inf; 1.2; 3.0; -0.4];
rewMat(1,3) = -0.6;
rewMat(2,4) = -0.56;
rewMat(3,5) = -0.46;
rewMat(4,6) = -0.62;
rewMat(5,7) = -0.55;
nT = 5;
m = size(rewMat,2) - nT;

hypos = [1,2,1,3,4,5];
hyposCard = [2,2,1,1];
clusters = [1,2,3,4];
clustersCard = [2,2];
nC = length(clustersCard);

intoThis = [1,2];

begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
begsH = tCloud2BegInd(hyposCard);
endsH = tCloud2EndInd(hyposCard);

probLogHypos = log(0.5*ones(1,4));
%probLogHypos = log(1:4);

% First I want to do brute force calcuation of normalization constant

iC = 1;
assocLocal = [1,1;1,0];
gainMatPostC = [rewMat; eye(2), -Inf*ones(2,5)];
gainMatPostC(gainMatPostC == 0) = -Inf;
nHypoTotalMax = 100;


preTN1 = ~isinf(gainMatPostC);
preTN2 = find(vec(preTN1));
trackNumberLookup = NaN*ones(size(gainMatPostC));
trackNumberLookup(preTN2) = preTN2;
k = 1;
indicesOfNewbornTracks = preTN2(end-1:end)';

gainMatPostC(isinf(gainMatPostC)) = -1000;

% Let's set this low to try to see the effects of recycling
% In this example there are 28 posterior hypotheses in total
nHypoTotalMax = 5;

[hyposLocal,hyposCardLocal,probLogLocal,kInvestigate,pqLen,priorCardAve,pq] ...
    = branchAndBoundExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,...
    iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoTotalMax,trackNumberLookup,k);

num_tracks = nT;
num_measurements = m;
[LBP_marginals, bethe_loglikelihood] = lbp_mex(gainMatPostC, num_tracks, num_measurements, hypos, hyposCard, probLogHypos, clusters, clustersCard);



% We need to compute two things - The normalization constant for the
% recycling proportion, and then the Murty marginals normalized by Bethe

% The normalization constants are easy
Z_murty = exp(logsumexp(probLogLocal, 2));
Z_bethe = exp(bethe_loglikelihood);
Z_P = Z_bethe - Z_murty;

% The marginals can be computed by looping over all hypotheses found by
% Murty and adding the score found to the tracks that exist 
% This is how we can achieve this - We loop over each hypothesis and uses
% trackNumberLookup to backtrack what association was made to create that
% track. We then add the score to the proper element in the marginals
% matrix. For tracks that does not appear in the 

murty_marginals = zeros(size(LBP_marginals));

begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
begsH = tCloud2BegInd(hyposCardLocal);
endsH = tCloud2EndInd(hyposCardLocal);

nH = length(hyposCardLocal);

new_tracks = trackNumberLookup(:);
new_tracks = new_tracks(~isnan(new_tracks));

%bethe_loglikelihood = logsumexp(vertcat(pq.score), 1);

% Loop over all posterior hypotheses
% Find out what tracks exist and what don't
% Add the hypothesis score to either the association or to nonexistence
for iH = 1:nH
    tracks_not_in_hypo = 1:num_tracks;
    tracks_in_hypo = hyposLocal(begsH(iH):endsH(iH));
    old_tracks = map_new_tracks_to_old_tracks(tracks_in_hypo, trackNumberLookup);
    tracks_not_in_hypo(old_tracks) = [];
    hypo_prob = exp(probLogLocal(iH) - bethe_loglikelihood);
    for track = tracks_in_hypo
        [t,j] = find(trackNumberLookup == track);
        % If the track is made from misdetection, map it to 1
        if j > num_measurements
            j = 1;
        else
            % We need to offset by 1 if it was detection
            j = j + 1;
        end
        murty_marginals(j, t) = murty_marginals(j, t) + hypo_prob;
    end
    for track = tracks_not_in_hypo
        murty_marginals(num_measurements + 2, track) = murty_marginals(num_measurements + 2, track) + hypo_prob;
    end
end

% Now that we have the Murty marginals, compute the recycling weights
p_P = LBP_marginals - murty_marginals;



function original_tracks = map_new_tracks_to_old_tracks(new_tracks, trackNumberLookup)
    original_tracks = zeros(1, length(new_tracks));
    k = 1;
    for new_track = new_tracks
        [t, ~] = find(trackNumberLookup == new_track);
        original_tracks(k) = t;
        k = k + 1;
    end

    original_tracks = unique(original_tracks);
end





