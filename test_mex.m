load('./data/at612/priorLikelihood612.mat');

% lbp_mex(gainMatPostC, num_tracks, num_measurements, hypos, hyposCard, probLogHypos, clusters, clustersCard);
num_tracks = size(trackFile,2);
num_measurements = size(measurements,2);


o = lbp_mex(gainMatPostC, num_tracks, num_measurements, hypos, hyposCard, probLogHypos, clusters, clustersCard);