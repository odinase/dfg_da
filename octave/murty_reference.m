function murty_reference(out_dir, ks, file_list)
% MURTY_REFERENCE  Octave/MATLAB reference output for the Python Murty port.
%
%   murty_reference(out_dir, ks, file_list)
%
% Runs the original MATLAB Murty pipeline -- branchAndBoundExplore.m per supercluster,
% newborn-hypothesis fixup, trackProbAccumulatePureOdin + margs2distrs, per-cluster
% logsumexp -- over every .mat listed in file_list, once per truncation depth in ks.
%
% This is a parameterised trim of compute_results.m; the two local functions below are
% copied from it verbatim, because the point of this harness is to compare the Python
% port against the original code, not against a rewrite of it.
%
% For each file/K it saves, into out_dir/nHypoTotalMax_<K>/<name>.mat:
%   a, Z                                    -- the end-to-end marginals + norm. constant
%   bbHyposLocal, bbHyposCardLocal,
%   bbProbLogLocal, nSuperclusters          -- the raw branchAndBoundExplore output per
%                                              supercluster, so a parity failure can be
%                                              localised to the branch-and-bound rather
%                                              than to the marginal accumulation.
%
% Run from the repo root:
%   octave --no-gui --quiet --eval "addpath('pmbm-cm-matlab', 'pmbm-cm-matlab/at612/jmpdFunctions', 'octave'); murty_reference('octave_murty_reference', [10 20 50 100 150], 'octave/reference_files.txt')"

  fid = fopen(file_list, 'r');
  if fid < 0
    error('cannot open file list %s', file_list);
  end
  filenames = {};
  tline = fgetl(fid);
  while ischar(tline)
    t = strtrim(tline);
    if ~isempty(t)
      filenames{end+1} = t;
    end
    tline = fgetl(fid);
  end
  fclose(fid);

  num_files = length(filenames);

  if ~isfolder(out_dir)
    mkdir(out_dir);
  end

  for K = ks(:)'
    folderPath = fullfile(out_dir, sprintf('nHypoTotalMax_%d', K));
    if ~isfolder(folderPath)
      mkdir(folderPath);
    end

    for i = 1:num_files
      filename = filenames{i};
      fprintf('K=%d  [%d/%d]  %s\n', K, i, num_files, filename);
      w = load(filename);

      hypos         = w.hypos;
      hyposCard     = w.hyposCard;
      clusters      = w.clusters;
      clustersCard  = w.clustersCard;
      probLogHypos  = w.probLogHypos;
      assocLocal    = w.assocLocal;
      gainMatPostC  = w.gainMatPostC;
      indicesOfNewbornTracks = w.indicesOfNewbornTracks;
      trackNumberLookup      = w.trackNumberLookup;
      meaHistColNew = w.meaHistColNew;
      k             = w.k;

      masters = assocLocal(1, assocLocal(2, :) == 1);
      num_measurements = size(w.measurements, 2);

      nHypoTotalMax = K;

      newHypos     = zeros(1,0);
      newHyposCard = zeros(1,0);
      newProbLogs  = zeros(1,0);
      cWNew        = zeros(1,0);
      cCardWNew    = zeros(1,0);
      newHyposCount = 0;

      tfClusterMembership = NaN*zeros(1, size(meaHistColNew, 2));

      nSuperclusters = size(masters, 2);
      bbHyposLocal     = cell(1, nSuperclusters);
      bbHyposCardLocal = cell(1, nSuperclusters);
      bbProbLogLocal   = cell(1, nSuperclusters);

      for iC = 1:nSuperclusters
        [hyposLocal, hyposCardLocal, probLogLocal] = branchAndBoundExplore( ...
            hypos, hyposCard, clusters, clustersCard, probLogHypos, iC, assocLocal, ...
            gainMatPostC, indicesOfNewbornTracks, nHypoTotalMax, trackNumberLookup, k);

        bbHyposLocal{iC}     = hyposLocal;
        bbHyposCardLocal{iC} = hyposCardLocal;
        bbProbLogLocal{iC}   = probLogLocal;

        newHypos     = [newHypos, hyposLocal];
        newHyposCard = [newHyposCard, hyposCardLocal];
        newProbLogs  = [newProbLogs, probLogLocal];

        newHyposOld = newHyposCount;
        newHyposCount = newHyposCount + size(hyposCardLocal, 2);
        cWNew = [cWNew, (newHyposOld+1):newHyposCount];
        cCardWNew = [cCardWNew, size((newHyposOld+1):newHyposCount, 2)];
        tracksInCluster = unique(hyposLocal);
        tfClusterMembership(tracksInCluster) = iC;
      end

      m = size(indicesOfNewbornTracks, 2);
      for jj = 1:m
        tIndex = indicesOfNewbornTracks(jj);
        if(~ismember(jj, meaHistColNew(end, newHypos)))
          hI = tIndex;
          newHyposCard = [newHyposCard, 1];
          newHypos = [newHypos, hI];
          newProbLogs = [newProbLogs, log(1)];
          hypoNumber = size(newHyposCard, 2);
          tCluster = tfClusterMembership(tIndex);
          if(isnan(tCluster))
            cWNew = [cWNew, length(newHyposCard)];
            cCardWNew = [cCardWNew, 1];
          else
            [cWNew, cCardWNew] = insertElements(tCluster, hypoNumber, cWNew, cCardWNew, 1);
          end
        end
      end

      hypos        = newHypos;
      hyposCard    = newHyposCard;
      clusters     = cWNew;
      clustersCard = cCardWNew;
      probLogHypos = newProbLogs;

      num_tracks_new = sum(sum(~isnan(trackNumberLookup)));
      [margs, probabilitiesCell] = trackProbAccumulatePureOdin( ...
          num_tracks_new, hypos, hyposCard, clusters, clustersCard, probLogHypos);
      a = margs2distrs(margs, trackNumberLookup, num_measurements);

      begsC = tCloud2BegInd(clustersCard);
      endsC = tCloud2EndInd(clustersCard);
      num_clusters = length(clustersCard);
      log_Zc = zeros(1, num_clusters);
      for iC = 1:num_clusters
        log_Zc(iC) = logsumexp(probLogHypos(begsC(iC):endsC(iC)), 2);
      end
      logZ = sum(log_Zc);
      Z = exp(logZ);

      [~, f, ext] = fileparts(filename);
      outfile = fullfile(folderPath, [f, ext]);
      save('-v7', outfile, 'Z', 'a', 'bbHyposLocal', 'bbHyposCardLocal', ...
           'bbProbLogLocal', 'nSuperclusters');
    end
  end
end


% ---------------------------------------------------------------------------------
% Local functions, copied verbatim from compute_results.m:165-248
% ---------------------------------------------------------------------------------

function distrs = margs2distrs(margs, trackNumberLookup, num_measurements)

    num_tracks = size(trackNumberLookup, 1) - num_measurements;
    distrs = zeros(num_tracks, num_measurements + 2);

    nT = size(margs, 2);


    for t = 1:nT
        [tt, j] = find(trackNumberLookup == t);
        if tt > num_tracks
            continue;
        end
        p = sum(margs{t});
        is_misdetection = j > num_measurements;
        if is_misdetection
            distrs(tt, 1) = distrs(tt, 1) + p;
        else
            distrs(tt, j + 1) = distrs(tt, j + 1) + p;
        end
    end

    distrs(:, end) = 1 - sum(distrs(:, 1:end-1), 2);
end


function [margs, probabilitiesCell] = trackProbAccumulatePureOdin(nTracks, hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHyposWill)

% New version of trackProbAccumulate that does not use meaHistCol
% Written by Edmund Brekke, starting 3rd of July 2020.

probabilitiesCell = probLogs2Probabilities(probLogHyposWill,clustersWill,clustersCardWill);

trackSumProbs = zeros(1,nTracks);
trackTotalProbs = zeros(1,nTracks);
clusterPerTrack = zeros(1,nTracks);
hyposPerTrack = cell(1,nTracks);

margs = cell(1, nTracks);

for t=1:nTracks

    [clusterNumbers,hyposCol,probabilities] = track2Cluster(t,hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHyposWill);

    if(length(unique(clusterNumbers)) > 1)
        error('several clusters assigned to one track');
    end
    clusterPerTrack(t) = unique(clusterNumbers);

    if(length(hyposCol) == 1)
        hyposPerTrack(t) = hyposCol;
    else
        hyposPerTrack{t} = cell2mat(hyposCol);
    end

    if(~isempty(clusterNumbers) && ~isnan(clusterNumbers(1)))

        % Need to convert a-level hypothesis numbers to b-level

        hList = hyposCol{1};
        inClustersA = [];
        for ii=1:size(hList,2)

            inClustersA = [inClustersA,find(clustersWill == hList(ii))];
        end

        [b,c] = a2bcFaster(inClustersA,clustersCardWill);

        probabilitiesThis = probabilitiesCell{clusterNumbers};
        margs{t} = probabilitiesThis(b);
        trackSumProbs(t) = sum(probabilitiesThis(b));
    end
end
end
