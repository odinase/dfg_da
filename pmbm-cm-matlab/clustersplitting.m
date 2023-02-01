function [hyposNew,hyposNewCard,clustersNew,clustersNewCard,probabilitiesAS,splittingTracks] = clustersplitting(hypos,hyposCard,clusters,clustersCard,probLogHypos,trackFile,trackFileShadow,meaHistCol,inCol,splittingThreshold)

% Function to carry out cluster splitting (CS) in PMBM implementation
% Written by Edmund Brekke during April 2020 - February 2020
% @hypos        Track numbers group together in the hypotheses
% @hyposCard    Cardinality array of hypos
% @clusters     Hypothesis numbers grouped together in clusters
% @clustersCard Cardinality array of clusters
% @probLogHypos     Logaritm of probabilities for all hypotheses
% @trackFile        Track file containing state estimates, covariances etc.
% @meaHistCol       List of all tracks as sequences of measurements
% @inCol:           Struct that describes the verical organization of trackFile
% @splittingThreshold:  Just a threshold
% >hyposNew         Track numbers group as hypotheses after CS
% >hyposNewCard     Cardinality array of hyposNew
% >clustersNew      Hypothesis numbers as clusters after CS
% >clustersNewCard  Cardinality array of clustersNew
% >probabilitiesAS  Probabilities of hypotheses after CS
% >splittingTracks  Tracks that have been identified for removal as part of CS
% $track2Cluster    Function to identify cluster and sum of hypothesis probabilities for each track
% $probLogs2Probabilities   
% $covVec2Mat

% Main principle: To split up excessively large clusters from Double Murty
% we remove tracks which both have low total track probabilities (TTP) and 
%which (potentially and actually) contribute to agglomeration.

% Calculate TTP for all tracks

nTracks = size(meaHistCol,2);
trackProbs = zeros(1,size(meaHistCol,2));
for ii=1:size(meaHistCol,2)
    [~,~,pI] = track2Cluster(ii,hypos,hyposCard,clusters,clustersCard,probLogHypos);
    trackProbs(ii) = pI*trackFile(inCol.exi,ii);
end

% Analyze number of recent consecutive misdetections
% ... because tracks with many recent misdetections are good candidates for removal

lastConsecutiveMisdets = zeros(1,nTracks);
for ii=1:nTracks
    counter = 0;
    for jj=1:size(meaHistCol,1)
        if(meaHistCol(end+1-jj,ii) ~= 0)
            break;
        else
            counter = counter + 1;
        end
    end
    lastConsecutiveMisdets(ii) = counter;
end

% Identify the tracks that shall split clusters

splittingTracks = find(lastConsecutiveMisdets > 1 & trackProbs < splittingThreshold);
cOfSplittingTracks = track2Cluster(splittingTracks,hypos,hyposCard,clusters,clustersCard);

% Prepare for splitting loop and preallocate for new hypothesis collection.

probabilitiesCell = probLogs2Probabilities(probLogHypos,clusters,clustersCard);
cBegs = tCloud2BegInd(clustersCard);
cEnds = tCloud2EndInd(clustersCard);
hBegs = tCloud2BegInd(hyposCard);
hEnds = tCloud2EndInd(hyposCard);
clustersNew = NaN*zeros(1,size(clusters,2));
clustersNewCard = NaN*zeros(1,size(clusters,2));
hyposNew = NaN*zeros(1,size(hypos,2));
hyposNewCard = NaN*zeros(1,size(hyposCard,2));
probabilitiesAS = NaN*zeros(1,size(probLogHypos,2));
pointerH = 0;
pointerHC = 0;
pointerC = 0;
pointerCC = 0;
nC = length(clustersCard);

%--------------------------------------------------------------------------
% Splitting loop begins
%--------------------------------------------------------------------------

for c=1:nC
    
    % I have already found splitting tracks, but which of these are in cluster c?
    
    splittingInC = splittingTracks(cOfSplittingTracks == c);
    
    % Then I also need to obtain the complete list of tracks in cluster c
    
    tracksInC = cluster2Tracks(c,hypos,hyposCard,clusters,clustersCard);
    activeTracks = setdiff(tracksInC,splittingInC);
    nTA = size(activeTracks,2);
    
    % Then its time to run clustering on activeTracks
    % Create a matrix OmegaS so that I just can use the function clusterTracks?
    
    omegaS = zeros(nTA,nTA);
    for t=1:nTA
        % Find all tracks that are within "validation gate" of tracksInC(t)
        
        xT = trackFile(inCol.tarX,activeTracks(t));
        pT = covVec2Mat(trackFile(inCol.tarP,activeTracks(t)));
        % Use only position or full state vector here?
        for jj=setdiff(1:nTA,t)
            xJ = trackFile(inCol.tarX,activeTracks(jj));
            %pJ = covVec2Mat(trackFile(inCol.tarP,activeTracks(jj)));
            maha = (xT-xJ)'*(pT\(xT-xJ));
            if(maha < 4^2)
                omegaS(t,jj) = 1;
            end
            
            % Also look at claimed measurements here

            meaNowT = meaHistCol(end,activeTracks(t));
            meaNowJ = meaHistCol(end,activeTracks(jj));
            if(meaNowT == meaNowJ && meaNowT > 0)
                omegaS(t,jj) = 1;
            end
        end
    end
    
    omegaS = (omegaS+omegaS') > 0;
    assocLocal = zeros(2,nTA);  % Array for checking which tracks are to be clustered together
    assocLocal(2,:) = assocLocal(2,:)*NaN;
    for p=1:nTA   % Iterate through all tracks
        if(isnan(assocLocal(2,p)))
            assocLocal(2,p) = p;
            assocLocal(1,p) = 1;
        end
        for i=[1:p-1,p+1:nTA]   % Iterate through the other tracks
            if(omegaS(p,i) > 0 )
                if(assocLocal(2,i) > assocLocal(2,p))
                    slaveIndices = find(assocLocal(2,:) == assocLocal(2,i));
                    assocLocal(1,slaveIndices) = 0;
                    assocLocal(2,slaveIndices) = assocLocal(2,p);
                elseif(assocLocal(2,i) < assocLocal(2,p))
                    slaveIndices = find(assocLocal(2,:) == assocLocal(2,p));
                    assocLocal(1,slaveIndices) = 0;
                    assocLocal(2,slaveIndices) = assocLocal(2,i);
                else
                    assocLocal(2,i) = assocLocal(2,p);
                end
            end
        end
    end
    
    % Now I basically have the new clusters. How to extract them and work with them?
    % First it would be nice to have all hypotheses in old cluster c
    
    hyposInC = clusters(cBegs(c):cEnds(c));
    masters = find(assocLocal(1,:));
    nDC = size(masters,2); % Number of separate sub-clusters (to become new and finer clusters)
    probabilitiesC = probabilitiesCell{c};
    
    for d=1:nDC  % Looping through the new clusters for a given parent cluster
        
        activeInD = activeTracks(assocLocal(2,:) == masters(d));
        heapHypos = zeros(1,0); % New hypotheses for this particular new cluster
        heapHyposCard = zeros(1,0);
        heapHyposProbs = zeros(1,0);

        for iH=1:size(hyposInC,2)
            h = hypos(hBegs(hyposInC(iH)):hEnds(hyposInC(iH)));
            
            % Then I need to pick active and inactive tracks in h
            
            activeInH = intersect(activeInD,h);
            eaBegs = tCloud2BegInd(heapHyposCard);
            eaEnds = tCloud2EndInd(heapHyposCard);
            testInHeap = false;
            eaNumbers = zeros(1,0);
            for ea=1:length(heapHyposCard)
                hEA = heapHypos(eaBegs(ea):eaEnds(ea));
                if((~isempty(activeInH) && size(hEA,2) == size(activeInH,2) && all(hEA == activeInH)) || (isempty(activeInH) && size(hEA,2) == 0))
                    testInHeap = true;
                    eaNumbers = [eaNumbers,ea];
                else
                    
                end
            end
            if(length(eaNumbers) > 1)
                error('Should not be more than one hypo in heap that matches current hypo');
            end
            if(~testInHeap)
                heapHypos = [heapHypos,activeInH];
                heapHyposCard = [heapHyposCard,size(activeInH,2)];
                heapHyposProbs = [heapHyposProbs,probabilitiesC(iH)];
            else
                heapHyposProbs(eaNumbers) = heapHyposProbs(eaNumbers) + probabilitiesC(iH);
            end
        end

        % Expand the post-CS hypothesis collection with contributions from current daughter cluster
        
        hyposNew(:,(pointerH+1):(pointerH+size(heapHypos,2))) = heapHypos;
        pointerH = pointerH + size(heapHypos,2);
        hyposNewCard(:,(pointerHC+1):(pointerHC+size(heapHyposCard,2))) = heapHyposCard;
        probabilitiesAS(:,(pointerHC+1):(pointerHC+size(heapHyposCard,2))) = heapHyposProbs;
        pointerHC = pointerHC + size(heapHyposCard,2);
        clustersNew(:,(pointerC+1):(pointerC+size(heapHyposCard,2))) = pointerC+(1:size(heapHyposCard,2));
        pointerC = pointerC + size(heapHyposCard,2);
        clustersNewCard(pointerCC+1) = size(heapHyposCard,2);
        pointerCC = pointerCC + 1;

        
    end
end

%--------------------------------------------------------------------------
% Splitting loop ends
%--------------------------------------------------------------------------

% Remove preallocated array contents that were not used.

hyposNew(:,(pointerH+1):end) = [];
hyposNewCard(:,(pointerHC+1):end) = [];
clustersNew(:,(pointerC+1):end) = [];
probabilitiesAS(:,(pointerHC+1):end) = [];
clustersNewCard(:,(pointerCC+1):end) = [];

