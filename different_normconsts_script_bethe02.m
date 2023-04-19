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
nHypoTotalMax = 20;


preTN1 = ~isinf(gainMatPostC);
preTN2 = find(vec(preTN1));
trackNumberLookup = NaN*ones(size(gainMatPostC));
trackNumberLookup(preTN2) = preTN2;
k = 1;
indicesOfNewbornTracks = preTN2(end-1:end)';

gainMatPostC(isinf(gainMatPostC)) = -1000;

% Let's set this low to try to see the effects of recycling
% In this example there are 28 posterior hypotheses in total
nHypoTotalMax = 100;

[hyposLocal,hyposCardLocal,probLogLocal,kInvestigate,pqLen,priorCardAve,pq] ...
    = branchAndBoundExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,...
    iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoTotalMax,trackNumberLookup,k);

num_tracks = nT;
num_measurements = m;
[LBP_marginals, bethe_loglikelihood] = lbp_mex(gainMatPostC, num_tracks, num_measurements, hypos, hyposCard, probLogHypos, clusters, clustersCard);



% We need to compute two things - The normalization constant for the
% recycling proportion, and then the Murty marginals normalized by Bethe

% The normalization constants are easy
Z_bethe = exp(bethe_loglikelihood);

Zs_murty = cumsum(exp(probLogLocal));
x = 1:length(Zs_murty);
plot(x, Zs_murty)
hold on
plot(x, Z_bethe*ones(1,length(x)))

legend('Zs murty', 'Z bethe')


