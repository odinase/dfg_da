function [hyposNew,hyposCardNew,clustersNew,clustersCardNew,probLogsNew,meaHistColNew,trackFileNew,trackFileShadowNew] ...
    = nScanMerge(hypos,hyposCard,clusters,clustersCard,probLogHypos,meaHistCol,inCol,trackFile,trackFileShadow)

% Function for n-scan track merging (NSM) in PMBM with cluster management
% Written by Edmund Brekke spring/summer 2021
% @hypos        Track numbers group together in the hypotheses
% @hyposCard    Cardinality array of hypos
% @clusters     Hypothesis numbers grouped together in clusters
% @clustersCard Cardinality array of clusters
% @probLogHypos     Logaritm of probabilities for all hypotheses
% @trackFile        Track file containing state estimates, covariances etc.
% @meaHistCol       List of all tracks as sequences of measurements
% @inCol:           Struct that describes the verical organization of trackFile
% @trackFileShadow: Track file with third dimension representing sliding window
% >hyposNew         Track numbers group as hypotheses after NSM
% >hyposCardNew     Cardinality array of hyposNew
% >clustersNew      Hypothesis numbers as clusters after NSM
% >clustersCardNew  Cardinality array of clustersNew
% >probLogsNew      Probabilities of hypotheses after NSM
% >meaHistColNew
% >trackFileNew
% >trackFileShadow
% $repeatedTracks:  Constructs master-slavel relationships for repeated tracks in meaHistCol
% $trackProbAccumulatePure: Gives total track probabilities (TTP's) which I use as mixture weights
% $covVec2Mat:      Convert vector representation to matrix representation of covariance matrix.
% $covMat2Vec:      Convert covariance matrix to its vector representation
% $gmReduce:        Generic Gaussian mixture reduction function
% $probLogs2Probabilities: Converts logarithms of probabilities into cell array of probabilities, on cell for each cluster.
% $pickIndC:        ABC bookeeping function used to pick track collections per hypothesis
% $checkSameTrackInDifferentClusters: Included for debugging purposes only.


% -------------------------------------------------------------------------
% Single linkage track clustering and re-indexing
% -------------------------------------------------------------------------

% Proximity criterion: Having the same column in meaHistCol

maxLag = size(meaHistCol,1);
[repeatedIx] = repeatedTracks(meaHistCol);

% repeatedIx has two rows:
% Row 1 contains the index of the master claim the track according to single-linkage clustering with the above proximity criterion.
% Row 2 contains a 1 for all master tracks, 0 for all claimed tracks.


% Find the new track numbers of all tracks:
% First get the right numbers for master tracks. Slave tracks do not yet have their correct numbers.
% Then get the right numbers also for slave tracks by looking up their masters.

newTrackNumbers = cumsum(repeatedIx(2,:),2);
slaveTrackBool = repeatedIx(2,:) == 0;
newTrackNumbers(slaveTrackBool) = newTrackNumbers(repeatedIx(1,slaveTrackBool));

% -------------------------------------------------------------------------
% Mixture reduction for each master track
% -------------------------------------------------------------------------

% I use TTP as mixture weigths.
% Not yet sure if this is the same as what Fontana, Garcia and Maskell use in their Bernoulli merging paper.

[~,ttp,~] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);

trackFileNew = zeros(size(trackFile,1),newTrackNumbers(end));
meaHistColNew = zeros(maxLag,newTrackNumbers(end));
trackFileShadowNew = zeros(size(trackFileShadow,1),newTrackNumbers(end),maxLag);

for ii=1:newTrackNumbers(end)
    
    firstI = find(newTrackNumbers == ii,1,'first');
    everyI = find(repeatedIx(1,:) == firstI);
    
    
    statesPreMix = trackFile(inCol.tarX,everyI);
    covsPreMix = covVec2Mat(trackFile(inCol.tarP,everyI));
    
    % Use TTP as mixture weights for now.
    
    weightsPreMix = ttp(everyI);
    weightsPreMix = weightsPreMix/sum(weightsPreMix);
    [etaAve,covAve] = gmReduce(statesPreMix,covsPreMix,weightsPreMix);
    trackFileNew(inCol.tarX,ii) = etaAve;
    trackFileNew(inCol.tarP,ii) = covMat2Vec(covAve);
    
    % Average existence probability
    
    exiPreMix = trackFile(inCol.exi,everyI);
    ttpWeighting = ttp(everyI)';
    ttpWeighting = ttpWeighting/sum(ttpWeighting);
    trackFileNew(inCol.exi,ii) = exiPreMix*ttpWeighting;
    
    % For several metadata fields in inCol averages are not meaningful - Just use track with maximal TTP.
    
    [temp,strongest] = max(ttp(everyI));
    trackFileNew(inCol.label,ii) = trackFile(inCol.label,everyI(strongest));
    trackFileNew(inCol.last,ii) = trackFile(inCol.last,everyI(strongest));
    trackFileNew(inCol.meaLast,ii) = trackFile(inCol.meaLast,everyI(strongest));
    trackFileNew(inCol.visi,ii) = trackFile(inCol.visi,everyI(strongest));
    trackFileNew(inCol.cost,ii) = trackFile(inCol.cost,everyI(strongest));
    trackFileNew(inCol.contrib,ii) = trackFile(inCol.contrib,everyI(strongest));
    meaHistColNew(:,ii) = meaHistCol(:,everyI(strongest));
    
    % Do the same mixture reduction things also for trackFileShadowNew
    
    for lag=1:maxLag
        statesPreMix = trackFileShadow(inCol.tarX,everyI,lag);
        covsPreMix = covVec2Mat(trackFileShadow(inCol.tarP,everyI,lag));
        [etaAve,covAve] = gmReduce(statesPreMix,covsPreMix,weightsPreMix);
        trackFileShadowNew(inCol.tarX,ii,lag) = etaAve;
        trackFileShadowNew(inCol.tarP,ii,lag) = covMat2Vec(covAve);
        exiPreMix = trackFileShadow(inCol.exi,everyI,lag);
        trackFileShadowNew(inCol.exi,ii,lag) = exiPreMix*ttpWeighting;
        trackFileShadowNew(inCol.label,ii,lag) = trackFileShadow(inCol.label,everyI(strongest),lag);
        trackFileShadowNew(inCol.last,ii,lag) = trackFileShadow(inCol.last,everyI(strongest),lag);
        trackFileShadowNew(inCol.meaLast,ii,lag) = trackFileShadow(inCol.meaLast,everyI(strongest),lag);
        trackFileShadowNew(inCol.visi,ii,lag) = trackFileShadow(inCol.visi,everyI(strongest),lag);
        trackFileShadowNew(inCol.cost,ii,lag) = trackFileShadow(inCol.cost,everyI(strongest),lag);
        trackFileShadowNew(inCol.contrib,ii,lag) = trackFileShadow(inCol.contrib,everyI(strongest),lag);
    end
end


% -------------------------------------------------------------------------
% Make the new hypothesis and cluster collections consiste of post-merge tracks
% -------------------------------------------------------------------------

% Basic ABC bookkeeping beginning and end indices

cBegs = tCloud2BegInd(clustersCard);
cEnds = tCloud2EndInd(clustersCard);
hBegs = tCloud2BegInd(hyposCard);
hEnds = tCloud2EndInd(hyposCard);

% Initialize new containers for hypotheses, clusters and logarithmic probabilities

hyposNew = zeros(1,0);
hyposCardNew = zeros(1,0);
clustersNew = zeros(1,0);
clustersCardNew = zeros(1,0);
probLogsNew = zeros(1,0);

% Calculate the probabilities of all old hypotheses
% (Clearly done before to get the TTPs, so I guess this call could be skipped with some revisions)

probabilitiesCell = probLogs2Probabilities(probLogHypos,clusters,clustersCard);

for c=1:size(clustersCard,2)
    
    % Pick the hypotheses and probabilities in cluster c
    
    hyposInC = clusters(cBegs(c):cEnds(c));
    probabilitiesC = probabilitiesCell{c};
    
    % The while loop below works on the collection of hypotheses until all have been accounted for.
    % This can also be seen as an instance of single-linkage clustering between hypotheses...
    % ... with proximity criterion given in the if-clause below.
    
    
    heapInC = hyposInC; % The collection of hypotheses that I'm working on. The while loop goes until this collection has been emptied.
    heapProbs = probabilitiesC;
    
    % Initialize new collection of hypotheses/probabilities for cluster c.
    
    hNewC = zeros(1,0);
    probLogsNewC = zeros(1,0);
    
    % Iterate until heapInC is emptied.
    
    iter = 1;
    maxIter = 1e5;
    while(~isempty(heapInC))
        master = heapInC(1); % Hypothesis that we currently are working on
        masterTracks = hypos(hBegs(master):hEnds(master)); % All tracks in that hypothesis
        masterTracksN = newTrackNumbers(masterTracks); % The same tracks with the new indexing
        
        comb = 1; % Contains hypotheses to be merged - indexing relative to heapInC (which is why it starts with 1).
        heapKeep = true(size(heapInC));
        heapKeep(1) = false;
        for ii=2:size(heapInC,2)
            candidate = heapInC(ii);
            candidateTracks = hypos(hBegs(candidate):hEnds(candidate));
            candidateTracksN = newTrackNumbers(candidateTracks);
            if(size(masterTracksN,2) == size(candidateTracksN,2) && all(masterTracksN == candidateTracksN,2) || (isempty(masterTracksN) && isempty(candidateTracksN)))
                comb = [comb,ii];
                heapKeep(ii) = false;
            end
        end
        
        newProb = sum(heapProbs(comb));
        probLogsNewC = [probLogsNewC,log(newProb)];
        hNewC = [hNewC,master];
        heapInC = heapInC(heapKeep);
        heapProbs = heapProbs(heapKeep);
        
        iter = iter + 1;
    end
    
    clustersCardNew = [clustersCardNew,size(hNewC,2)];
    hNewContent = pickIndC(hNewC,hyposCard);
    hyposNew = [hyposNew,newTrackNumbers(hypos(hNewContent))];
    hyposCardNew = [hyposCardNew,hyposCard(hNewC)];
    probLogsNew = [probLogsNew,probLogsNewC];
    clustersNew = 1:sum(clustersCardNew);
    
end

% Finally some debugging test because I had a problem with repeated tracks

[testBool,clustersProblematic] = checkSameTrackInDifferentClusters(1:size(trackFileNew,2),hyposNew,hyposCardNew,clustersNew,clustersCardNew);
if(any(testBool))
    
    % New version with hypothesis pruning
    
    hyposKillBool = false(1,size(hyposCardNew,2));
    [probabilities] = probLogs2ProbabilitiesForHypos(1:sum(clustersCardNew),probLogsNew,clustersNew,clustersCardNew,sum(clustersCardNew));
    problematicTracks = find(testBool);
    for jj=1:length(problematicTracks)
        tt = problematicTracks(jj);
        meaSeq = meaHistColNew(:,tt);
        
        [tracks,hyposCol,clusterNumbers] = findTrackAllAllowClusterShare(meaSeq,meaHistColNew,hyposNew,hyposCardNew,clustersNew,clustersCardNew);
        cn = unique(clusterNumbers);
        probMasses = zeros(1,length(cn));
        for ii=1:length(cn)
            clusterI = cn(ii);
            
            % What is the total probability mass claimed by hypotheses containing meaSeq in this cluster?
            
            hyposI = hyposCol(clusterNumbers == clusterI);
            probMasses(ii) = sum(probabilities(hyposI));
        end
        [temp,bestClusterIndex] = max(probMasses);
        bestCluster = cn(bestClusterIndex);
        hyposI = hyposCol(clusterNumbers == bestCluster);
        hyposOther = setdiff(hyposCol,hyposI); % Hypotheses that we want to kill
        
        % Shall I just mark these for deletion and delete all hypotheses in one go?
        % Seems much tidier than this gradual reduction of the hypothesis collection
        
        hyposKillBool(hyposOther) = true;
        


        
    end
    
    hRemove = find(hyposKillBool);
    
    nTNew = size(meaHistColNew,2);
    [hyposNew,hyposCardNew,clustersNew,clustersCardNew,probLogsNew,new2OldTrackNumbers,old2NewTrackNumbers] = removeHypos(hRemove,hyposNew,hyposCardNew,clustersNew,clustersCardNew,probLogsNew,nTNew);
    
    meaHistColNew = meaHistColNew(:,new2OldTrackNumbers);
    trackFileNew = trackFileNew(:,new2OldTrackNumbers);
    trackFileShadowNew = trackFileShadowNew(:,new2OldTrackNumbers,:);
    
    
    %error('rwerw');
    
    
    
    % Old version
    
    
    
    %
    %     [probabilities] = probLogs2ProbabilitiesForHypos(1:sum(clustersCardNew),probLogsNew,clustersNew,clustersCardNew,sum(clustersCardNew));
    %
    %     % In this case only keep the cluster which contains the best hypothesis.
    %
    %     assocLocal = zeros(2,size(clustersCard,2));
    %
    %     %assocLocal = [1:size(clustersCardNew,2); zeros(1,size(clustersCardNew,2))];
    %
    %     for ii=1:length(clustersProblematic)
    %         cp = sort(unique(clustersProblematic{ii}),'ascend');
    %         ifTest = ~any(ismember(cp,assocLocal(1,:)));
    %         if(~any(ismember(cp,assocLocal(1,:))))
    %
    %             % New cluster-connection
    %
    %             assocLocal(1,cp) = cp(1);
    %             assocLocal(2,cp(1)) = 1;
    %             assocLocal(2,cp(2:end)) = 0;
    %         else
    %             assocLocal(1,cp) = cp(1);
    %         end
    %     end
    %     for ii=1:size(assocLocal,2)
    %         if(assocLocal(1,ii) == 0)
    %             assocLocal(1,ii) = ii;
    %         end
    %     end
    %
    %assocLocal
    
    %error('kkk');
    %
    %     masters = find(assocLocal(2,:) == 1);
    %     iter = 1;
    %     while(~isempty(masters) && iter < 100)
    %
    %         iter = iter + 1;
    %         clustersInvolved = find(assocLocal(1,:) == masters(1));
    %
    %         % Here I could use something like pickIndC?
    %
    %         [aRemove,tCloudRemove] = pickIndC(clustersInvolved,clustersCardNew);
    %         probsRelevant = probabilities(aRemove);
    %         [temp,bestHypoAmongThese] = max(probsRelevant);
    %         [b,bestCluster] = a2bc(aRemove(bestHypoAmongThese),clustersCardNew);
    %
    %         clustersToPrune = setdiff(clustersInvolved,bestCluster);
    %
    %         %[indicesRemaining,tCloudNew] = removeC(c,tCloud)
    %
    %         [aRemove,tCloudRemove,aRemain,tCloudRemain] = pickIndC(clustersToPrune,clustersCardNew);
    %         [hyposRemaining1,clustersCardPostPrune] = removeC(clustersToPrune,clustersCardNew); % So far all looks good
    %         [hEntriesRemaining,hyposCardPostPrune] = removeC(aRemove,hyposCardNew); % hyposCardPostPrune also looks sensible
    %         hyposPostPrune = hyposNew(hEntriesRemaining); % hyposPostPrune contains remaining track indices. Looks also reasonable.
    %
    %         [hyposNew,hyposCardNew,meaHistColNew,trackFileNew,trackFileShadowNew] = pruningAdjustHypos(hyposPostPrune,hyposCardPostPrune,meaHistColNew,trackFileNew,trackFileShadowNew);
    %
    %         [boolsH,boolsT] = testRepeatedMeasurements(hyposNew,hyposCardNew,meaHistColNew);
    %         if(any(boolsH))
    %             error('Did I get repeated measurements in a single hypothesis inside NSM?');
    %         end
    %         clustersNew = 1:sum(clustersCardPostPrune);
    %         clustersCardNew = clustersCardPostPrune;
    %
    %
    %         [testBool,clustersProblematic] = checkSameTrackInDifferentClusters(1:size(trackFileNew,2),hyposNew,hyposCardNew,clustersNew,clustersCardNew);
    %         if(any(testBool))
    %             error('still got repeated measurement in loop');
    %         end
    %
    %
    %         % Revise aLCopy part
    %
    %         aLCopy = assocLocal;
    %         aLCopy(1,clustersToPrune) = 0;
    %         aLCopy(2,bestCluster) = 0; % Done with this one now
    %
    %         aLCopy2 = [zeros(1,size(aLCopy,2));aLCopy(2,:)];
    %
    %         for jj=1:size(aLCopy,2)
    %
    %             %             if(jj==3)
    %             %                error('kkjj');
    %             %             end
    %
    %             if(aLCopy(1,jj) > 0 && aLCopy2(1,jj) == 0)
    %                 x = sum(aLCopy(1,1:jj) > 0);
    %                 aLCopy2(1,aLCopy(1,:) == aLCopy(1,jj)) = x;
    %
    %             end
    %
    %         end
    %         %aLCopy
    %         aLCopy2(:,aLCopy(1,:) == 0) = [];
    %
    %
    %         if(any(aLCopy2(1,:) > 1:size(aLCopy2,2)))
    %             error('to high entry value in aLCopy');
    %         end
    %
    %
    %         assocLocal = aLCopy2;
    %         masters = find(assocLocal(2,:) == 1);
    %
    %     end
end



[testBool,clustersProblematic] = checkSameTrackInDifferentClusters(1:size(trackFileNew,2),hyposNew,hyposCardNew,clustersNew,clustersCardNew);
if(any(testBool))
    error('still got repeated measurement!');
end