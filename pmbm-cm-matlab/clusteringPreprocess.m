function [assocLocal,gainMatPostC,masters] = clusteringPreprocess(hypos,hyposCard,clusters,clustersCard,gainMatFull,meaHistCol,m,preClusterThreshold)


begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
nT = size(meaHistCol,2);
lMatFull = gainMatFull(1:nT,1:m);


% -------------------------------------------------------------------------
% Find connected components of cluster-measurement graph
% -------------------------------------------------------------------------



nC = size(clustersCard,2);
lMatBool = ~isinf(lMatFull);
ix = find(lMatBool(:))';
subs = v2m(ix,nT);
trackNumbers = subs(1,:);
meaNumbers = nC + subs(2,:);  % Adding nC because measurements are nodes placed after the cluster nodes in node list

% I need to go from track numbers to cluster numbers

clusterNumbers = track2Cluster(trackNumbers,hypos,hyposCard,clusters,clustersCard);



if(m > 0 && any(~isinf(lMatFull(:))))
    
    % Construct edge list, value list and the graph itself.
    
    %tempvals = lMatFull(ix)';
    tempvals = reshape(lMatFull(ix),fliplr(size(ix)));
    [tempvals2,sia] = sort(tempvals,'descend');
    cm = [clusterNumbers(sia);meaNumbers(sia)];
    [edgearr,temp] = unique(cm','rows');
    edgearr = edgearr';
    temp = temp';
    vals = tempvals2(temp)';
    misDetPart = diag(gainMatFull(1:nT,m+1:end));
    trackwiseThreshold = misDetPart - preClusterThreshold; % Asssignments less likely than misdetection / 500 are to be removed IF they connect cluster.
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
    
    % Prepare what I can before the while loop:
    % What are the maximal cardinalities and counts for each cluster
    
    maxCardClusters = zeros(1,nC);
    maxCountClusters = zeros(1,nC);
    for ii=1:nC
        iC = begsC(ii):endsC(ii);
        maxCardClusters(ii) = max(hyposCard(iC));
        maxCountClusters(ii) = size(iC,2);
    end
    maxCountHeuri = max(maxCountClusters) + 100;
    maxCardHeuri = max(maxCardClusters) + 5;
    
    % Then build knowledge about worst-edges to break
    
    bins = conncomp(H);
    clusterAssoc = zeros(1,max(meaNumbers));
    for ii=1:length(unique(bins))
        q = find(bins==ii);
        clusterAssoc(q) = q(1);
    end
    clusterAssoc = clusterAssoc(1:nC);
    worstBreakEdgeGains = -Inf*ones(nC,nC);
    breakEdgeStuff = cell(nC,nC,7); % Contains measurements, tracks for C1, tracks for C2, new reward submatrix, winning clusters for each measurement, a_match,b_match
    for ii=1:nC
        q1 = find(clusterAssoc==clusterAssoc(ii));
        q = q1(q1 > ii);
        if(length(q) >= 1)
            for jI=1:length(q)
                jj = q(jI);
                
                % Here I should investigate all edges that go
                % between measurements shared by both cluster ii and
                % cluster jj.
                
                % Exploit that H.EdgesEndNodes exhibits the bipartite nature of the cluster-to-measurement graph
                
                edgesIBool = H.Edges.EndNodes(:,1) == ii;
                edgesJBool = H.Edges.EndNodes(:,1) == jj;
                
                meaI = H.Edges.EndNodes(edgesIBool,2)-nC;
                meaJ = H.Edges.EndNodes(edgesJBool,2)-nC;
                meaCommon = intersect(meaI,meaJ)';
                
                winningClusters = zeros(size(meaCommon));
                
                if(~isempty(meaCommon))
                    
                    % For each common measurement, identify all gating tracks in both clusters
                    
                    tracksInI = cluster2Tracks(ii,hypos,hyposCard,clusters,clustersCard);
                    tracksInJ = cluster2Tracks(jj,hypos,hyposCard,clusters,clustersCard);
                    
                    
                    breakEdgeStuff{ii,jj,1} = meaCommon;
                    breakEdgeStuff{ii,jj,2} = tracksInI;
                    breakEdgeStuff{ii,jj,3} = tracksInJ;
                    
                    [tracksUnion,a_match,b_match] = union_sorted(tracksInI,tracksInJ); % Notice that I use union_sorted from lightspeed library.
                    
                    rewSub = gainMatFull(tracksUnion,meaCommon);  % But is it really the original reward matrix I should use her?
                    
                    
                    for zz=1:size(meaCommon,2)
                        
                        %zCol = gainMatFull(1:nT,meaCommon(zz));
                        zCol = rewSub(:,zz);
                        tracksGatingZ = find(zCol > -Inf); % Track indices relative to the union
                        tracksIZ = intersect(a_match,tracksGatingZ); % Track indices relative to the union
                        tracksJZ = intersect(b_match,tracksGatingZ); % Track indices relative to the union
                        
                        % Identify winning cluster for mea zz
                        % On which side do we have the worst edge to cut?
                        
                        rewIZ = rewSub(tracksIZ,zz);
                        maxLikeliI = max(rewIZ);
                        rewJZ = rewSub(tracksJZ,zz);
                        maxLikeliJ = max(rewJZ);
                        
                        if(maxLikeliI > maxLikeliJ)
                            winningClusters(zz) = ii;
                            rewJZ = -Inf*ones(size(rewJZ));
                            rewSub(tracksJZ,zz) = rewJZ;
                            worstBreakEdgeGains(ii,jj) = max(worstBreakEdgeGains(ii,jj),maxLikeliJ);
                        else
                            winningClusters(zz) = jj;
                            rewIZ = -Inf*ones(size(rewIZ));
                            rewSub(tracksIZ,zz) = rewIZ;
                            worstBreakEdgeGains(ii,jj) = max(worstBreakEdgeGains(ii,jj),maxLikeliI);
                        end
                    end
                    breakEdgeStuff{ii,jj,4} = rewSub;
                    breakEdgeStuff{ii,jj,5} = winningClusters;
                    breakEdgeStuff{ii,jj,6} = a_match;
                    breakEdgeStuff{ii,jj,7} = b_match;
                end
            end
        end
    end
    
    % Initialize collection of edges that may be broken
    
    edgesCand = zeros(2,0);
    for ii=1:nC
        for jj=1:nC
            if(~isempty(breakEdgeStuff{ii,jj,1}))
                edgesCand = [edgesCand,[ii;jj]];
            end
            
        end
    end
    
    % Initialize collection of broken edges
    
    edgesBroken = zeros(2,0);
    clustersAcceptable = false;
    nycaCount = 1;
    while(~clustersAcceptable && nycaCount < 1200)
        nycaCount = nycaCount+1;
        bins = conncomp(H);
        clusterAssoc = zeros(1,max(meaNumbers));
        for ii=1:length(unique(bins))
            q = find(bins==ii);
            clusterAssoc(q) = q(1);
        end
        clusterAssoc = clusterAssoc(1:nC);
        [GC,GR] = groupcounts(clusterAssoc');
        uca = unique(clusterAssoc);
        cardSumsPerSupercluster = zeros(size(uca));
        
        % Get the cardinality sums for each super-cluster
        
        
        violators = false(1,nC);
        
        
        % Test whether superclusters are sufficiently small
        
        testSuperclusterSize = ~any(GC > 5);
        if(~testSuperclusterSize)
            superclusterViolators = find(GC > 5);
            for ii=1:length(superclusterViolators)
                clusterViolators = find(clusterAssoc ==  uca(superclusterViolators(ii)));
                violators(clusterViolators) = true;
            end
        end
        
        % Test whether sums of max-cardinalities are sufficiently small
        
        cardMaxTests = false(size(uca));
        for ii=1:length(uca)
            maxCardI = maxCardClusters(clusterAssoc == uca(ii));
            cardMaxTests(ii) = sum(maxCardI) > maxCardHeuri & length(maxCardI) > 1;
            if(cardMaxTests(ii))
                a = find(clusterAssoc == uca(ii));
                violators(a) = true;
            end
        end
        testSuperclusterMaxTotalCard = ~any(cardMaxTests);
        
        % Test whether sums of hypothesis counts are sufficiently small
        
        countMaxTests = false(size(uca));
        for ii=1:length(uca)
            countI = maxCountClusters(clusterAssoc == uca(ii));
            countMaxTests(ii) = sum(countI) > maxCountHeuri & length(countI) > 1;
            if(countMaxTests(ii))
                a = find(clusterAssoc == uca(ii));
                violators(a) = true;
            end
        end
        testSuperclusterMaxTotalCount = ~any(countMaxTests);
        
        if(testSuperclusterSize && testSuperclusterMaxTotalCard && testSuperclusterMaxTotalCount)
            clustersAcceptable = true;
        else
            
            % First calculate heuristic for relevant cluster-cluster-connections
            
            heuristicList = zeros(1,size(edgesCand,2));
            for ii=1:size(edgesCand,2)
                sizeContrib = (sum(clusterAssoc == clusterAssoc(edgesCand(1,ii))))/5;
                cardContrib = (maxCardClusters(edgesCand(1,ii)) + maxCardClusters(edgesCand(2,ii)))/maxCardHeuri;
                countContrib = (maxCountClusters(edgesCand(1,ii)) + maxCountClusters(edgesCand(2,ii)))/maxCountHeuri;
                edgeContrib = worstBreakEdgeGains(edgesCand(1,ii),edgesCand(2,ii))*3/4;
                
                heuristicList(ii) = sizeContrib + cardContrib + countContrib - edgeContrib;
            end
            
            % Decide which cluster-custer-connection to break
            
            [~,ix] = max(heuristicList);
            clustersToCut = edgesCand(:,ix);
            edgesBroken = [edgesBroken,clustersToCut];
            edgesCand(:,ix) = [];
            
            % Update gainMatFull accordingly
            
            rewSub = breakEdgeStuff{clustersToCut(1),clustersToCut(2),4};
            tracksUnion = union(breakEdgeStuff{clustersToCut(1),clustersToCut(2),2},breakEdgeStuff{clustersToCut(1),clustersToCut(2),3});
            meaCommon = breakEdgeStuff{clustersToCut(1),clustersToCut(2),1};
            gainMatFull(tracksUnion,meaCommon) = rewSub;
            
            
            % Remove edges to losing cluster for each affected measurement in clustering graph
            
            winnerStuff = breakEdgeStuff{clustersToCut(1),clustersToCut(2),5};
            for jj=1:size(meaCommon,2)
                winnerJ = winnerStuff(jj);
                loserJ = setdiff(clustersToCut,winnerJ);
                hee =  H.Edges.EndNodes';
                ixe = hee(1,:) == loserJ & hee(2,:) == nC+meaCommon(jj);
                H = rmedge(H,find(ixe));
            end
        end
    end
    
    [~,ia] = setdiff(edgearr',H.Edges.EndNodes,'rows');
    toBeBroken(ia) = 1;
    
    
    % I make an edited copy of the reward matrix to ensure that it conforms
    % with the edge removals done above.
    
    gainMatPostC = gainMatFull;
    lMatPostC = gainMatPostC(1:nT,1:m);
    
    % Remove all assignments in lMatCopy that lack cluster support
    
    for ii=find(toBeBroken)
        c = edgearr(1,ii);
        jj = edgearr(2,ii) - nC;
        h = clusters(begsC(c):endsC(c));
        [aRemove,tCloudRemove,aRemain,tCloudRemain] = pickIndC(h,hyposCard);
        t = unique(hypos(aRemove));
        for tt=1:length(t)
            lMatPostC(t(tt),jj) = -Inf;
        end
    end
    gainMatPostC(1:nT,1:m) = lMatPostC;
    
elseif(m > 0 && all(isinf(lMatFull(:))))
    
    % In this case I have measurements, but none of them are validated
    
    gainMatPostC = gainMatFull;
    lMatPostC = gainMatPostC(1:nT,1:m);
    clusterAssoc = 1:nC;
    
else
    gainMatPostC = gainMatFull;
    lMatPostC = gainMatPostC(1:nT,1:m);
    clusterAssoc = 1:nC;
end


[x,ia] = unique(clusterAssoc);
masters = clusterAssoc(ia);

assocLocal = zeros(2,size(clusterAssoc,2));
assocLocal(1,:) = clusterAssoc;
assocLocal(2,masters) = 1;