clear all;
% Specify the path to the text file
file_path = './files_used.txt';
output_folder = './murty_output';

% Open the text file
fid = fopen(file_path, 'r');

% Initialize an empty cell array to store the filenames
filenames = {};

% Read each line of the file until the end
tline = fgets(fid);
while ischar(tline)
    % Store the line (filename) in the cell array
    filenames = [filenames; strtrim(tline)];
    
    % Read the next line
    tline = fgets(fid);
end

% Close the file
fclose(fid);

num_files = length(filenames);
iMC = 0;

hypo_params = [10, 20, 50, 100, 150];  % List of numbers

for K = hypo_params
    folderPath = fullfile('.', sprintf('murty_output/nHypoTotalMax_%d', K));    
    
    if ~isfolder(folderPath)
        mkdir(folderPath);
        disp(['Directory ', folderPath, ' created successfully.']);
    else
        disp(['Directory ', folderPath, ' already exists.']);
    end

    

for i = 1:num_files
    filename = filenames{i};
    fprintf("Doing file %s!\n", filename);
    load(filename);
    begsC = tCloud2BegInd(clustersCard);
    endsC = tCloud2EndInd(clustersCard);
    begsH = tCloud2BegInd(hyposCard);
    endsH = tCloud2EndInd(hyposCard);
    masters = assocLocal(1, assocLocal(2, :) == 1);
    
    nT = size(trackFile,2);
    m = size(measurements,2);
    
    num_tracks = nT;
    num_measurements = m;
    
    newHypos = zeros(1,0); % To contain track numbers for each of the new hypotheses after clustering
    newHyposCard = zeros(1,0);
    newProbLogs = zeros(1,0);
    cWNew = zeros(1,0);
    cCardWNew = zeros(1,0);
    newHyposCount = 0;

    tfClusterMembership = NaN*zeros(1,size(meaHistColNew,2));  % Which new cluster a track belongs to
  
%     if nHypoTotalMax ~= 150
%        nHypoTotalMax
%        filename
%     end
    nHypoTotalMax = K;
    fprintf("Starting Murty with nHypoTotalMax = %i!\n", nHypoTotalMax);

    for iC=1:size(masters,2)
    
        [hyposLocal,hyposCardLocal,probLogLocal,kInvesti,pqLen,priorCardAve,pq] = branchAndBoundExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoTotalMax,trackNumberLookup,k);
        
        newHypos = [newHypos,hyposLocal];
        newHyposCard = [newHyposCard,hyposCardLocal];
        newProbLogs = [newProbLogs,probLogLocal];
        
        
        newHyposOld = newHyposCount;
        newHyposCount = newHyposCount + size(hyposCardLocal,2);
        cWNew = [cWNew,(newHyposOld+1):newHyposCount];
        cCardWNew = [cCardWNew,size((newHyposOld+1):newHyposCount,2)];
        clusterNumber = iC;
        tracksInCluster = unique(hyposLocal);
        tfClusterMembership(tracksInCluster) = clusterNumber;
        
    end

    fprintf("Murty done!\n");

    for jj=1:m
        tIndex =indicesOfNewbornTracks(jj);  % Number of current track
        
        % Is this track a member of any hypothesis?
        % If not, then we need to add it as a separate hypothesis
        
        %if(~ismember(tIndex,newHypos))
        if(~ismember(jj,meaHistColNew(end,newHypos))) % New criterion only allows separate newborn cluster if MEASUREMENT not claimed in other clusters
            hI = tIndex;
            newHyposCard = [newHyposCard,1];
            newHypos = [newHypos,hI];
            newProbLogs = [newProbLogs,log(1)];
            hypoNumber = size(newHyposCard,2);
            tCluster = tfClusterMembership(tIndex);
            if(isnan(tCluster))
                cWNew = [cWNew,length(newHyposCard)];
                cCardWNew = [cCardWNew,1];
            else
                [cWNew,cCardWNew] = insertElements(tCluster,hypoNumber,cWNew,cCardWNew,1);
            end
        end
    end

    hypos = newHypos;
    hyposCard = newHyposCard;
    clusters = cWNew;
    clustersCard = cCardWNew;
    probLogHypos = newProbLogs;
    
    fprintf("Computing marginals\n");
    num_tracks_new = sum(~isnan(trackNumberLookup),'all');
    [margs, probabilitiesCell] = trackProbAccumulatePureOdin(num_tracks_new, hypos,hyposCard,clusters,clustersCard,probLogHypos);
    a = margs2distrs(margs, trackNumberLookup, num_measurements);
    fprintf("Computing marginals done!\n");


    fprintf("Computing normalization constants\n");
    begsC = tCloud2BegInd(clustersCard);
    endsC = tCloud2EndInd(clustersCard);
    
    num_clusters = length(clustersCard);
    
    log_Zc = zeros(1, num_clusters);
    
    for iC = 1:num_clusters

        bC = begsC(iC);
        eC = endsC(iC);
    
        log_Zc(iC) = logsumexp(probLogHypos(bC:eC), 2);

    end

    logZ = sum(log_Zc);
    Z = exp(logZ);
    fprintf("Computing normalization constants done!\n");

    [~, f, ext] = fileparts(filename);
    outfile = [folderPath, '/', f, ext];
    save(outfile, 'Z', 'a', '-v4');


    fprintf("Done! Has completed %i out of %i files (%.3f %%)\n", i, num_files, i / num_files * 100.0);
end

end

% delete(gcp('nocreate'));


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
    %meaSeq = meaHistCol(:,t);
    %[tracks,hyposCol,clusterNumbers] = findTrack(meaSeq,meaHistCol,hyposWill,hyposCardWill,clustersWill,clustersCardWill);
    
    [clusterNumbers,hyposCol,probabilities] = track2Cluster(t,hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHyposWill);
    
    
    % Must use an alternative to find track
    
    
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

































function [M, murty_const] = murty_marginals(num_tracks, num_measurements, hyposLocal, trackNumberLookup, probLogLocal, hyposCardLocal)

log_murty_norm = logsumexp(probLogLocal, 2);

begsH = tCloud2BegInd(hyposCardLocal);
endsH = tCloud2EndInd(hyposCardLocal);

nH = length(hyposCardLocal);

for iH = 1:nH
    tracks_not_in_hypo = 1:num_tracks;
    tracks_in_hypo = hyposLocal(begsH(iH):endsH(iH));
    tracks_in_hypo(tracks_in_hypo > num_tracks) = []; % Is this correct?? We might look up newborn tracks... Delete?
    old_tracks = map_new_tracks_to_old_tracks(tracks_in_hypo, trackNumberLookup);
    tracks_not_in_hypo(old_tracks) = [];
    hypo_prob = exp(probLogLocal(iH) - log_murty_norm);
    M = zeros(num_measurements + 2, num_tracks);
    for track = tracks_in_hypo
        [t,j] = find(trackNumberLookup == track);
        % If the track is made from misdetection, map it to 1
        if j > num_measurements
            j = 1;
        else
            % We need to offset by 1 if it was detection
            j = j + 1;
        end
        M(j, t) = M(j, t) + hypo_prob;
    end
    for track = tracks_not_in_hypo
        M(num_measurements + 2, track) = M(num_measurements + 2, track) + hypo_prob;
    end
end

murty_const = exp(log_murty_norm);

end


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