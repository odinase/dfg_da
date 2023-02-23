hypos = [1, 2, 1, 3, 4, 5];
hyposCard = [2, 2, 1, 1];
probLogHypos = [log(0.5), log(0.5), log(0.5), log(0.5)];
clusters = [1, 2, 3, 4];
clustersCard = [2, 2];
num_tracks = 5;
num_measurements = 2;
gainMatFull = -Inf*rand(num_tracks + num_measurements, num_tracks + num_measurements);
gainMatFull(1:num_tracks, :) = [...
            3.0, -inf,   -0.60, -inf, -inf, -inf, -inf;...
            3.2, -inf, -inf,   -0.56, -inf, -inf, -inf;...
           -3.0,     1.2, -inf, -inf,   -0.46, -inf, -inf;...
        -inf,     3.0, -inf, -inf, -inf,   -0.62, -inf;...
        -inf,    -0.4, -inf, -inf, -inf, -inf,   -0.55...
    ];


k = 1;
indicesOfNewbornTracks = (num_tracks+1):(num_measurements + num_tracks);

j = 1;
for i = indicesOfNewbornTracks
    gainMatFull(i, j) = 0.0;
    j = j + 1;
end

preClusterThreshold = -inf;
m = num_measurements;
meaHistCol = ones(8,0);
trackNumberLookup = NaN*zeros(num_tracks + num_measurements, num_tracks + num_measurements);
for i=1:num_tracks
    trackNumberLookup(i,i) = i;
end
[assocLocal,gainMatPostC,masters]...
    = clusteringPreprocess(hypos,hyposCard,clusters,clustersCard,gainMatFull,meaHistCol,m,preClusterThreshold);
            

for iC=1:size(masters,2)
[hyposLocal,hyposCardLocal,probLogLocal,kInvesti,pqLen] ...
    = branchAndBoundExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,...
    iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoTotalMax,trackNumberLookup,k)
end