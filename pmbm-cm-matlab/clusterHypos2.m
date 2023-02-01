function [newHypos,newHyposCard,cWNew,cCardWNew,newProbLogs,assocLocal,gainMatPostC] = clusterHypos2(gainMatFull,proximities,labelShare,clustersWill,clustersCardWill,hyposWill,hyposCardWill,probLogHypos,k,trackFile,inCol,meaHistCol,m)

% -------------------------------------------------------------------------
% New version
% -------------------------------------------------------------------------

begsC = tCloud2BegInd(clustersCardWill);
endsC = tCloud2EndInd(clustersCardWill);
nTracks = size(meaHistCol,2);
lMatFull = gainMatFull(1:nTracks,1:m);


% -------------------------------------------------------------------------
% Find connected components of cluster-measurement graph
% -------------------------------------------------------------------------

nT = size(lMatFull,1);
nC = size(clustersCardWill,2);
lMatBool = ~isinf(lMatFull);
ix = find(lMatBool(:))';
subs = v2m(ix,nT);
trackNumbers = subs(1,:);
meaNumbers = nC + subs(2,:);  % Adding nC because measurements are nodes placed after the cluster nodes in node list

% I need to go from track numbers to cluster numbers

clusterNumbers = track2Cluster(trackNumbers,hyposWill,hyposCardWill,clustersWill,clustersCardWill);

% Construct edge list, value list and the graph itself. 

tempvals = lMatFull(ix)';
[tempvals2,sia] = sort(tempvals,'descend');
cm = [clusterNumbers(sia);meaNumbers(sia)];
[edgearr,temp] = unique(cm','rows');
edgearr = edgearr';
temp = temp';
vals = tempvals2(temp)';
misDetPart = diag(gainMatFull(1:nT,m+1:end));
trackwiseThreshold = misDetPart - 6.2; % Asssignments less likely than misdetection / 500 are to be removed IF they connect cluster.
G = graph(edgearr(1,:),edgearr(2,:),vals); % Be aware that the graph construction takes an unreasonable amount of run-time (0.2s)

% Centrality of the graph is used to mark edges that lie between clusters

wbc = centrality(G,'betweenness')';
connectingEdges = wbc(edgearr(2,:)) > 0;
thresholded = vals < trackwiseThreshold(edgearr(1,:))';
toBeBroken = thresholded & connectingEdges;

% Ensure that measurents that are totally liberated through this process 
% still aren assigned to their best fit cluster.

for jj=1:m
    jEdges = edgearr(2,:) == (jj+nC) & ~toBeBroken;
    if(~any(jEdges)) % Measurement jj has no edges that have not been broken -> Then it is totally liberated.
        relevant = find(edgearr(2,:) == (jj+nC));
        relvals = vals(relevant);
        [temp,ix] = max(relvals);
        bestEdge = relevant(ix);
        toBeBroken(bestEdge) = false;
    end
end

% Remove sufficiently weak connecting edges from cluster-measurement graph
% ... and find its connected components using inbuilt Matlab function.

H = rmedge(G,find(toBeBroken));
bins = conncomp(H);
clusterAssoc = bins(1:nC);

% I make an edited copy of the reward matrix to ensure that it conforms
% with the edge removals done above.

gainMatPostC = gainMatFull;
lMatPostC = gainMatPostC(1:nTracks,1:m);

% Remove all assignments in lMatCopy that lack cluster support 

for ii=find(toBeBroken)
    c = edgearr(1,ii);
    jj = edgearr(2,ii) - nC;
    h = clustersWill(begsC(c):endsC(c));
    [aRemove,tCloudRemove,aRemain,tCloudRemain] = pickIndC(h,hyposCardWill);
    t = unique(hyposWill(aRemove));
    for tt=1:length(t)
        lMatPostC(t(tt),jj) = -Inf;
    end
end
gainMatPostC(1:nTracks,1:m) = lMatPostC;

% -------------------------------------------------------------------------
% Perform Murty-based clustering based on reduced cluster-measurement graph
% -------------------------------------------------------------------------

[x,ia] = unique(clusterAssoc);
masters = clusterAssoc(ia);

assocLocal = zeros(2,size(clusterAssoc,2));
assocLocal(1,:) = clusterAssoc;
assocLocal(2,masters) = 1;
           
% Containers for new hypotheses and clusters

nClusters = length(masters);
newHypos = zeros(1,0); % To contain track numbers for each of the new hypotheses after clustering
newHyposCard = zeros(1,0);
newProbLogs = zeros(1,0);
cWNew = zeros(1,0);
cCardWNew = zeros(1,nClusters);

% Iterate through each of the new clusters

for iC=1:size(masters,2)
    intoThis = find(clusterAssoc == masters(iC)); % Find all clusters that need to be merged with original master iC
    contentThis = clustersWill(pickIndC(intoThis,clustersCardWill)); % This would be the hypotheses of to-be-clustered clusters.
    
    % Construct reward matrix to be used by clustering-purpose Murty
    
    firstContent = clustersWill(pickIndC(intoThis(1),clustersCardWill));
    firstScores = probLogHypos(firstContent);
    rewardFC = firstScores;
    for jC=2:length(intoThis)
        nextContent = clustersWill(pickIndC(intoThis(jC),clustersCardWill));
        nextScores = probLogHypos(nextContent);
        n1 = size(rewardFC,1);
        n2 = size(rewardFC,2);
        rewardFC = blkdiag(rewardFC,nextScores);
        rewardFC((n1+1):end,1:n2) = -Inf;
        rewardFC(1:n1,(n2+1):end) = -Inf;
    end
    %nHMaxInClustering = 30;
    nHMaxInClustering = max(100,max(clustersCardWill(intoThis)));
    
    if(all(isinf(rewardFC(:))))
        error('rewardFC is Inf');
    end
    
    % Run Murty
    
    [Customer2Item,Item2Customer,Nsolutions,Rewards]=MurtyRevised(rewardFC,nHMaxInClustering);

    % Construct the new hypothesis collection for each of the new clusters
    
    nHyposLast = size(newHyposCard,2);
    newHyposContentI = zeros(length(inCol.tarX)+length(inCol.tarP),0);
    newHyposMHI = zeros(size(meaHistCol,1),0);
    newHyposExi = zeros(1,0);
    newHyposContentICard = zeros(1,0);
    for jN=1:Nsolutions
        rowOfHypos = contentThis(Customer2Item(jN,:));
        newHypos = [newHypos,hyposWill(pickIndC(rowOfHypos,hyposCardWill))];  % What I add here is track labels
        newHyposCard = [newHyposCard,size(pickIndC(rowOfHypos,hyposCardWill),2)]; 
        newProbLogs = [newProbLogs,Rewards(jN)];
        newHyposContentI = [newHyposContentI,trackFile([inCol.tarX,inCol.tarP],hyposWill(pickIndC(rowOfHypos,hyposCardWill)))];
        newHyposContentICard = [newHyposContentICard, size(trackFile([inCol.tarX,inCol.tarP],hyposWill(pickIndC(rowOfHypos,hyposCardWill))),2)];
        newHyposMHI = [newHyposMHI,meaHistCol(:,hyposWill(pickIndC(rowOfHypos,hyposCardWill)))];
        newHyposExi = [newHyposExi,trackFile(inCol.exi,hyposWill(pickIndC(rowOfHypos,hyposCardWill)))];
    end
    nHyposNow = size(newHyposCard,2);
    begsHCI = tCloud2BegInd(newHyposContentICard);
    endsHCI = tCloud2EndInd(newHyposContentICard);
    cWNew = [cWNew,(nHyposLast+1):nHyposNow];
    cCardWNew(iC) = nHyposNow-nHyposLast;
    
    % Some error checking just in case
    
    if(any(cWNew > length(newHyposCard)))
        error('Got more hypotheses than I have hypothesis cardinalities');
    end
    if(any(cCardWNew < 0))
        error('Got negative cluster cardinalities');
    end
    [shareTracks,shareClusters] = testClustersShareTracks(cWNew,cCardWNew,newHypos,newHyposCard);
    if(~isempty(shareTracks))
        error('tracks shared among clusters inside clusterHyposWill');
    end
end

