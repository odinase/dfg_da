function [hyposLocal,hyposCardLocal,probLogLocal,kInvestigate,pqLen] = branchAndBoundExplore6(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoTotalMax,trackNumberLookup,k)


% Version 2: I must also block out measurements in other clusters
% To do this it would seem more natural to do the blocking out immediately after the expansion
% Version 3: Include the blocking out of rows / columns that actually should be part of Murty
% Version 4: Include aggressive switching when discrepancy between sum of local scores and global scores for switch front hypotheses.
% Version 6: Include loss-bound heuristic for the order of tracks

kInvestigate = zeros(1,0);
pqLen = 0;

minVal = -30;  % Not sure if this one actually should be used. Its a kind of pruning that perhaps is more suitable elsewhere.
superMinVal = -1000;
max_iter_auction = 5000;
m = size(indicesOfNewbornTracks,2);
nT = size(gainMatPostC,1)-m;
lMatPostC = gainMatPostC(1:nT,1:m);
masters = find(assocLocal(2,:) == 1);  % Placeholder to be realized when separated in function

gainMatPostC(isinf(gainMatPostC)) = superMinVal;

bestScoreInC = -Inf; % New dynamic parameter to be used for branch and bound purposes

intoThis = find(assocLocal(1,:) == masters(iC)); % Find all clusters that need to be merged with original master iC
%contentThis = clusters(pickIndC(intoThis,clustersCard)); % This would be the hypotheses of to-be-clustered clusters.
nCL = size(intoThis,2);  % Number of old clusters in local supercluster

begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
begsHypo = tCloud2BegInd(hyposCard);
endsHypo = tCloud2EndInd(hyposCard);

% Worked example tests

didAS = false;
didExpansion = false;
didExpansionStop = false;
didLS = false;
didLSStop = false;

% -------------------------------------------------------------------------
% Just rewrite the entire first phase
% First phase: Find best child hypothesis for each parent in each cluster
% -------------------------------------------------------------------------

bestLocal = struct('scores',{},'clustercontrib',{});
for ii=1:nCL
    parentHyposIndices = clusters(pickIndC(intoThis(ii),clustersCard)); % The indices of hypotheses in the cluster with index intoThis(ii)
    counterIH = 1; % To keep tracks of how many of the parent hypotheses in this cluster that actually are allowed offspring
    bestLocal(ii).scores = zeros(1,0);
    for iH=1:size(parentHyposIndices,2)
        tracksH = hypos(begsHypo(parentHyposIndices(iH)):endsHypo(parentHyposIndices(iH))); % The tracks involved in parent hypothesis number iH
        if(~isempty(tracksH))
            gatedH = find(any(~isinf(lMatPostC(tracksH,:)),1));
            meaIndH = [gatedH,m+tracksH]; % Indices of all measurements, including dummies, gated by tracks in H
            gainMatH = gainMatPostC(tracksH,meaIndH);
%             [personToItemCr,~,optRewardCr]=assign2D(-gainMatH); % Using David Crouses JVC-based assignment solver
%             optReward = - optRewardCr;
%             personToItem = personToItemCr';
            
            [personToItem,optReward,upperBoundsOfScores] = crouse2D(gainMatH); % upperBoundsOfScores is not used here. Perhaps something for Lars to look into.
            
            % Some alarm triggers in case assignment solver should do anything crazy
            
            if(isnan(sum(personToItem)))
                error('why NaN in assignment 1');
            end
            if(isempty(personToItem))
                error('why empty hypothesis');
            end
            
            % Store assignment results in struct that contains best new hypothesis for each parent hypothesis
            % New hypothesis desribed by meanIndOpt/meaIndOptLocal, while old hypothesis described by tracks
            
            bestLocal(ii).scores = [bestLocal(ii).scores,optReward + probLogHypos(parentHyposIndices(iH))];
            bestLocal(ii).clustercontrib(counterIH) = struct('tracks',tracksH,'meaIndAll',meaIndH,'meaIndOpt',meaIndH(personToItem),'meaIndOptLocal',personToItem,...
                'parentHypo',parentHyposIndices(iH),'parentPriorScore',probLogHypos(parentHyposIndices(iH)),'upperBoundsOfScores',upperBoundsOfScores);
            counterIH = counterIH + 1;
            
        else
            %  I have an empty hypothesis as parent. How do I handle this?
            
            bestLocal(ii).scores = [bestLocal(ii).scores,probLogHypos(parentHyposIndices(iH))];
            bestLocal(ii).clustercontrib(counterIH) = struct('tracks',zeros(1,0),'meaIndAll',zeros(1,0),'meaIndOpt',zeros(1,0),'meaIndOptLocal',zeros(1,0),...
                'parentHypo',parentHyposIndices(iH),'parentPriorScore',probLogHypos(parentHyposIndices(iH)),'upperBoundsOfScores',zeros(1,0));
            counterIH = counterIH + 1;
        end
    end
    [sortedScores,ix] = sort(bestLocal(ii).scores,'descend');
    bestLocal(ii).scores = sortedScores;
    bestLocal(ii).clustercontrib = bestLocal(ii).clustercontrib(ix);
end


if(~isempty(bestLocal))
    
    % -------------------------------------------------------------------------
    % Second phase: Branch-and-bound with cluster interactions when needed
    % -------------------------------------------------------------------------
    
    % Initialize priority queue
    
    labelHypo = 1;
    pq = pqInitialize(bestLocal,labelHypo,gainMatPostC,superMinVal); 
    
    if(~isempty(pq))
        
        % Initialize the depth first search 
        
        uub = pq(1).bound; % Changed from score previously
        lpb = superMinVal/2; % SOMEWHAT HEURISTIC - BUT DANGEROUS TO SIMPLY SET IT TO superMinVal
        uubHolder = 1;
        topScoreHolder = 1;
        iter = 1;
        maxIter = 10000;
        
        uubHist = uub;
        lpbHist = lpb;
        
        while(uub > lpb && iter <= maxIter) % Should also have a test for whether I actually have more candidates to investigate.
            iter = iter + 1;
            
            % Aggressive swithcing if necessary
            
            if(pq(uubHolder).score < pq(uubHolder).bound-1e-13) % Notice: Strict inequality
                
                % Identify contended measurements
                
                contentionClustersBool = false(1,nCL);
                for j1=1:nCL
                    for j2=(j1+1):nCL
                        parentInJ1 = pq(uubHolder).clustercontrib(j1).parentInBestList;
                        parentInJ2 = pq(uubHolder).clustercontrib(j2).parentInBestList;
                        if(~isempty(intersect(bestLocal(j1).clustercontrib(parentInJ1).meaIndOpt,bestLocal(j2).clustercontrib(parentInJ2).meaIndOpt)))
                            contentionClustersBool(j1) = true;
                            contentionClustersBool(j2) = true;
                        end
                    end
                end
                didAS = true;
                if(~any(contentionClustersBool))
                    error('There should not be a score gap if I have no contention clusters');
                end
                
                bestScoreAS = pq(uubHolder).score;
                
                unresolved = uubHolder; % List of hypotheses that possibly can have better switch-descendants
                leafNodes = uubHolder;
                
                sIter = 0;
                while(sIter < maxIter && ~isempty(unresolved))
                    sIter = sIter+1;
                    
                    % Pick top element among unResolved
                    
                    boundsUnresolved = vertcat(pq(unresolved).bound);
                    [~,ix] = max(boundsUnresolved);
                    psEntry = unresolved(ix); % This is pointer to Problem/Solution pair <P,S> in Miller's pseudo code
                    
                    
                    p = pq(psEntry); % Can I modify this during the for loop just like Miller modifies the problem P? The I can keep P out of pq
                    
                    % First find the actual best case loss values to determine order
                    
                    bclActual = -Inf*ones(1,nCL);
                    for ii=1:nCL
                        if(p.clustercontrib(ii).parentInBestList+1 <= length(bestLocal(ii).scores))
                            bclActual(ii) = p.bestCaseLoss(ii,p.clustercontrib(ii).parentInBestList+1)- ...
                                p.bestCaseLoss(ii,p.clustercontrib(ii).parentInBestList);
                        end
                    end
                    [svals,six] = sort(bclActual,'descend');                    
                    switchabilities = horzcat(p.clustercontrib.switchable);
                    asMurtyOrder = six(~isinf(svals) & p.bound + svals > p.score & switchabilities(six));
                    boundsLoop = -Inf*ones(size(asMurtyOrder)); % Negative Inf to enable me to take max of this later
                    
                    for ii=1:length(asMurtyOrder)
                        
                        yCluster = asMurtyOrder(ii); % This is y on line 4.3 in Miller's pseudo code.
                        zForY = p.clustercontrib(yCluster).parentInBestList;
                        pMarked = p;
                        pMarked.bestCaseLoss(yCluster,zForY) = -Inf; % Removing triple (line 4.3.2 in Miller et al)
                        
                        % Then I'm at line 4.3.3 finding solution. Do I need to use a full assign2D here?
                        % I can simply pick the earliest possible columns for each row
                        
                        [a,sMar] = max(pMarked.bestCaseLoss,[],2);
                        switchClustersBool = ~isinf(a) & sMar ~= vertcat(pMarked.clustercontrib.parentInBestList);
                        
                        if(any(switchClustersBool))
                            newParents = sMar(switchClustersBool);
                            switches = NaN*zeros(1,nCL);
                            switches(switchClustersBool) = newParents;
                            
                            subs = [1:nCL;horzcat(pMarked.clustercontrib.parentInBestList)];
                            for jj=1:length(switches)
                                if(~isnan(switches(jj)))
                                    subs(2,jj) = switches(jj);
                                end
                            end
                            lininds = m2v(subs,size(pMarked.bestCaseLoss));
                            losses = pMarked.bestCaseLoss(lininds);

                            if(sum(losses) >= lpb)
                                
                                t2mForbidden = zeros(1,0);
                                t2mEnforced = zeros(1,0);
                                [sMarked,labelHypo] = pqsolve(pMarked,bestLocal,switches,t2mForbidden,t2mEnforced,superMinVal,gainMatPostC,labelHypo,psEntry);
                                

                                pq(end+1) = sMarked;
                                pq(psEntry).childrenInSearchTree = [pq(psEntry).childrenInSearchTree,length(pq)];
                                pq(psEntry).childrenCount = pq(psEntry).childrenCount + 1;
                                
                                
                                if(any(~testBAndCpqI(sMarked)))
                                    error('Something weith wrong with B and C level indices');
                                end
                                
                                
                                
                                bestScoreAS = max(bestScoreAS,sMarked.score);
                                boundsLoop(ii) = sMarked.bound;
                                
                                % Remove parent from set of leaf nodes. Done inside if-clause because it would remain leaf node if it has no ancestors.
                                
                                leafNodes(leafNodes == psEntry) = [];
                                leafNodes = [leafNodes,length(pq)];
                                
                                % ---------------------------------------------
                                % Rewriting if(sMarked.bound > bestScoreAS) ....
                                % ---------------------------------------------
                                
                                if(sMarked.bound > pq(uubHolder).score)
                                    unresolved = [unresolved,length(pq)];
                                end
                                
                                unresolvedBounds = vertcat(pq(unresolved).bound);
                                toBeEliminated = unresolvedBounds < sMarked.score;
                                unresolved(toBeEliminated) = [];
                                
                            else
                                
                                % Something that got thresholded away - no worries: It must have had a bound less than LPB
                            end
                            
                            % Modify p so that y-z is enforced
                            
                            p.bestCaseLoss(yCluster,1:(zForY-1)) = -Inf;
                            p.bestCaseLoss(yCluster,(zForY+1):end) = -Inf;
                            
                        else
                            
                        end
                        pq(psEntry).clustercontrib(asMurtyOrder(ii)).switchable = false;
                    end
                    [bestBoundOfLoop,bestBoundHolder] = max(boundsLoop);
                    
                    
                    topNode = find(vertcat(pq.parentInSearchTree) ==0);

                    if(pq(uubHolder).bound > uub)
                        error('why bound increase');
                    end

                    unresLabels = vertcat(pq(unresolved).labelHypo);
                    temp = unresLabels == p.labelHypo;
                    unresolved(temp) = [];
                    
                    %unresolved(ix) = [];
                    
                    if(isnan(pq(end).bound))
                        error('why nan bound in AS 3');
                    end
                    scoreSortedList = sort(vertcat(pq.score),'descend');
                    
                    if(length(scoreSortedList) >= nHypoTotalMax)
                        lpb = max(lpb,scoreSortedList(nHypoTotalMax));
                    end
                    
                    if(uub > uubHist(end))
                        error('UUB just increased');
                    end
                    
                    uubHist = [uubHist,uub];
                    lpbHist = [lpbHist,lpb];
                end
                
                hLabels = horzcat(pq.labelHypo);
                number17 = find(hLabels == 17);
                number9 = find(hLabels == 9);
%                 if(~isempty(number17) && ~isempty(number9) && number9 > number17)
%                     error('How did L5 get past L4? Right after AS');
%                 end                  
%                 
%                 if(length(pq) == 17)
%                    error('stop at pq length 17'); 
%                 end
                % After exiting the while loop - sort pq
                
                [~,ixV] = sortrows([vertcat(pq.bound),vertcat(pq.score)],[-1,-2]);
                pq = pq(ixV);
                
                % Re-ordering of tree parents
                
                oldParents = vertcat(pq.parentInSearchTree);
                newParents = zeros(1,length(pq));
                for ii=1:length(pq)
                    newParents(oldParents==ii) = find(ixV==ii);
                end
                np = newParents(ixV);
                
                % Store re-ordered tree parents
                
                J=1:numel(ixV);
                J(ixV)=J;
                for ii=1:length(pq)
                    original = ixV(ii);
                    pq(ii).parentInSearchTree = newParents(J(original));
                end
                
                % I must also do re-ordering of tree children
                
                for ii=1:length(pq)
                    pq(ii).childrenInSearchTree = J(pq(ii).childrenInSearchTree);
                end
                uubHolder = J(uubHolder);
            end
            
                hLabels = horzcat(pq.labelHypo);
                number17 = find(hLabels == 17);
                number9 = find(hLabels == 9);
%                 if(~isempty(number17) && ~isempty(number9) && number9 > number17)
%                     error('How did L5 get past L4? Before Lazy switches');
%                 end                 
            
            
            switchabilityPossible = vertcat(pq(uubHolder).clustercontrib.switchable);
            p = pq(uubHolder);            
            
            % -------------------------------------------------------------
            % Lazy switches -----------------------------------------------
            % -------------------------------------------------------------

            % First find the actual best case loss values to determine order
            
            if(any(vertcat(p.clustercontrib.switchable)))
                bclActual = -Inf*ones(1,nCL);
                for ii=1:nCL
                        if(p.clustercontrib(ii).parentInBestList+1 <= length(bestLocal(ii).scores))
                            bclActual(ii) = p.bestCaseLoss(ii,p.clustercontrib(ii).parentInBestList+1)- ...
                                p.bestCaseLoss(ii,p.clustercontrib(ii).parentInBestList);
                        end
                end
                [svals,six] = sort(bclActual,'descend');
                
                % Define order of clusters for switches. Notice that the
                % requirement "p.bound + svals > p.score" used in AS is not
                % used here.
                
                switchabilities = horzcat(p.clustercontrib.switchable);
                asMurtyOrder = six(~isinf(svals) &  switchabilities(six));
                       
                
                
                %asMurtyOrder = six(~isinf(svals)  & horzcat(p.clustercontrib.switchable));
                boundsLoop = -Inf*ones(size(asMurtyOrder)); % Negative Inf to enable me to take max of this later
                for ii=1:length(asMurtyOrder)
                    
                    yCluster = asMurtyOrder(ii); % This is y on line 4.3 in Miller's pseudo code.
                    zForY = p.clustercontrib(yCluster).parentInBestList;
                    pMarked = p;
                    pMarked.bestCaseLoss(yCluster,zForY) = -Inf; % Removing triple (line 4.3.2 in Miller et al)
                    
%                             if(pq(uubHolder).labelHypo == 2)
%                                 
%                                 error('check LS of L2');
%                             end                    
                    
                    
                    % Then I'm at line 4.3.3 finding solution. Do I need to use a full assign2D here?
                    % I can simply pick the earliest possible columns for each row
                    
                    [a,sMar] = max(pMarked.bestCaseLoss,[],2);
                    switchClustersBool = ~isinf(a) & sMar ~= vertcat(pMarked.clustercontrib.parentInBestList);
                    
                    if(any(switchClustersBool))
                        newParents = sMar(switchClustersBool);
                        switches = NaN*zeros(1,nCL);
                        switches(switchClustersBool) = newParents;
                        
                        subs = [1:nCL;horzcat(pMarked.clustercontrib.parentInBestList)];
                        for jj=1:length(switches)
                            if(~isnan(switches(jj)))
                                subs(2,jj) = switches(jj);
                            end
                        end
                        lininds = m2v(subs,size(pMarked.bestCaseLoss));
                        losses = pMarked.bestCaseLoss(lininds);

                        if(sum(losses) >= lpb)
                            
                            didLS = true;
                            
                            t2mForbidden = zeros(1,0);
                            t2mEnforced = zeros(1,0);
                            [sMarked,labelHypo] = pqsolve(pMarked,bestLocal,switches,t2mForbidden,t2mEnforced,superMinVal,gainMatPostC,labelHypo,uubHolder);
                            
                            pq(end+1) = sMarked;
                            pq(uubHolder).childrenInSearchTree = [pq(uubHolder).childrenInSearchTree,length(pq)];
                            pq(uubHolder).childrenCount = pq(uubHolder).childrenCount + 1;                            
                            if(any(~testBAndCpqI(sMarked)))
                                error('Something weith wrong with B and C level indices');
                            end
                            
                            %bestScoreAS = max(bestScoreAS,sMarked.score);
                            boundsLoop(ii) = sMarked.bound;
                        else
                            
                            didLSStop = true;
                            
                            % Something that got thresholded away - no worries: It must have had a bound less than LPB
                        end

                        % Modify p so that y-z is enforced
                        
                        p.bestCaseLoss(yCluster,1:(zForY-1)) = -Inf;
                        p.bestCaseLoss(yCluster,(zForY+1):end) = -Inf;
                    else
                        
                    end
                    pq(uubHolder).clustercontrib(asMurtyOrder(ii)).switchable = false;
                end
                [bestBoundOfLoop,bestBoundHolder] = max(boundsLoop);
            end
            
            % -------------------------------------------------------------
            % Expansions --------------------------------------------------
            % -------------------------------------------------------------
  
            % New expansion for loop
            
            if(any(horzcat(p.clustercontrib.expandable)))
                
                % New loss bound calculation - should utilize p.upperBoundsOfScores
                
                scorebounds = -Inf*ones(size(p.tracksEntire));
                for tt=1:length(p.tracksEntire)
                    if(p.clustercontrib(p.tracksCLevel(tt)).expandable(p.tracksBLevel(tt)))
                       
                        tRow = p.upperBoundsOfScores(tt,:);
                        tRow(p.meaEntireOpt(tt)) = superMinVal;
                        scorebounds(tt) = max(tRow);     
                    end
                end
                [scoreBoundSorted,trackOrder] = sort(scorebounds,'ascend');

                %lossBoundSorted = lossbounds(trackOrder);
                for tt=1:length(trackOrder)
                    if(scoreBoundSorted(tt) > lpb)  % Test inside for-loop and not before in case I want to do any Miller-style stuff
                        didExpansion = true;
                        pMarked = p; % Miller line 4.3.1.
                        t2mForbidden = trackOrder(tt);
                        t2mEnforced = zeros(1,0);
                        switches = NaN*zeros(1,nCL);
                        parentInTree = uubHolder;
                        [pqO,labelHypo] = pqsolve2(pMarked,bestLocal,switches,t2mForbidden,t2mEnforced,superMinVal,gainMatPostC,labelHypo,parentInTree,minVal);
                                                 
                        if(~isempty(pqO))
                            if(pqO.bound >= lpb)
                                pq(end+1) = pqO;
                                pq(uubHolder).childrenInSearchTree = [pq(uubHolder).childrenInSearchTree,length(pq)];
                                pq(uubHolder).childrenCount = pq(uubHolder).childrenCount + 1;
                            else
                                labelHypo = labelHypo - 1;
                            end
                        end
                    else
                        didExpansionStop = true;
                        pq(uubHolder).clustercontrib(pq(uubHolder).tracksCLevel(tt)).expandable(pq(uubHolder).tracksBLevel(tt)) = false;
                    end
                    
                    % Enforce t in p before next for-iteration
                    
                    yForP = trackOrder(tt);
                    zForP = p.customer2Item(yForP);
                    p.rewardMatrix(1:(yForP-1),zForP) = superMinVal;
                    p.rewardMatrix((yForP+1):end,zForP) = superMinVal;
                    p.rewardMatrix(yForP,1:(zForP-1)) = superMinVal;
                    p.rewardMatrix(yForP,(zForP+1):end) = superMinVal;
                end
                for ii=1:nCL
                   pq(uubHolder).clustercontrib(ii).expandable = false*pq(uubHolder).clustercontrib(ii).expandable; 
                end
            end
            
            % Identity where UUB and LPB are now
            
            valueList =  vertcat(pq.bound);
            if(any(isnan(valueList)))
                error('why nan in valueList');
            end
            
            [newValueList,ixV] = sort(valueList,'descend');
            pq = pq(ixV);
            
            % Re-ordering of tree parents
            
            oldParents = vertcat(pq.parentInSearchTree);
            J=1:numel(valueList);
            J(ixV)=J;        
            JExt = [J,0];
            a = oldParents;
            a(a==0) = length(JExt);
            newParents = JExt(a);
            
            % Store re-ordered tree parents

            for ii=1:length(pq)
                original = ixV(ii);
                pq(ii).parentInSearchTree = newParents(J(original));
            end
            
            % I must also do re-ordering of tree children
            
            for ii=1:length(pq)
                pq(ii).childrenInSearchTree = J(pq(ii).childrenInSearchTree); % THIS STATEMENT IS ALSO CAUSING EXCESSIVE RUNTIME
            end
            
            % Before identifying UUB, identify LPB
            
            [scoreSortedList,ixScoreSorted] = sort(vertcat(pq.score),'descend');
            if(length(scoreSortedList) >= nHypoTotalMax)
                lpb = max(lpb,scoreSortedList(nHypoTotalMax));
            end
            % Where is UUB now?
            
            oldUUB = uub;
            
            hasBeenUUBList = vertcat(pq.hasBeenUUB);
            worseThanOldUUB = find((newValueList < uub + 100*eps) & ~hasBeenUUBList);
            if(~isempty(worseThanOldUUB) && uub > lpb)
                
                olduubHolder = uubHolder;
                uubHolder = worseThanOldUUB(1);
                
                pq(uubHolder).hasBeenUUB = true;
                olduub = uub;
                uub = newValueList(uubHolder);
                
            else
                break;
            end
            
            uubHist = [uubHist,uub];
            lpbHist = [lpbHist,lpb];


            
        end
        
        % Here I must remember to sort with respect to score!
        
        valueList =  vertcat(pq.score); % THIS TIME WRT. SCORE, NOT BOUND.
        [newValueList,ixV] = sort(valueList,'descend');
        pq = pq(ixV);
        % Re-ordering of tree parents
        
        oldParents = vertcat(pq.parentInSearchTree);
        J=1:numel(valueList);
        J(ixV)=J;
        JExt = [J,0];
        a = oldParents;
        a(a==0) = length(JExt);
        newParents = JExt(a);
        
        % Store re-ordered tree parents
        
        for ii=1:length(pq)
            original = ixV(ii);
            pq(ii).parentInSearchTree = newParents(J(original));
        end
        
        % I must also do re-ordering of tree children
        
        for ii=1:length(pq)
            pq(ii).childrenInSearchTree = J(pq(ii).childrenInSearchTree); % THIS STATEMENT IS ALSO CAUSING EXCESSIVE RUNTIME
        end
        
        % After hypothesis queue is score-sorted, proceed....
        
        pqLen = length(pq);
        finalHyposCount = min(length(pq),nHypoTotalMax);
        
        hInC = clusters(pickIndC(intoThis,clustersCard));
        tInC = hypos(pickIndC(hInC,hyposCard));
        gatedInC = find(any(~isinf(lMatPostC(tInC,:)),1));
        
        nTL = 0;
        for ii=1:size(intoThis,2)
            contentI = clusters(pickIndC(intoThis(ii),clustersCard));
            nTL = nTL + max(hyposCard(contentI));
        end
        nML = size(gatedInC,2);
        
        hLAlloc = (nTL+nML)*finalHyposCount;
        cLAlloc = finalHyposCount;
        pointerHL = 0;
        pointerCL = 0;
        
        hyposLocal = zeros(1,hLAlloc);
        hyposCardLocal = zeros(1,cLAlloc);
        probLogLocal = zeros(1,cLAlloc);
        
        for jj=1:finalHyposCount
            
            parentHypo = horzcat(pq(jj).clustercontrib.tracks);
            hNewLegacy = NaN*zeros(1,length(parentHypo));
            meaByLegacy = NaN*zeros(1,length(parentHypo));
            
            localCumCard = 0;
            for ii=1:nCL
                downIndicesI = pq(jj).clustercontrib(ii).tracks;
                alongIndicesI = pq(jj).meaEntire(pq(jj).clustercontrib(ii).meaIndOptLocalInSuper);
                %alongIndicesI = pq(jj).clustercontrib(ii).meaindices(pq(jj).clustercontrib(ii).hypos);
                
                for iL=1:length(downIndicesI)
                    hNewLegacy(1,localCumCard+iL) = trackNumberLookup(downIndicesI(iL),alongIndicesI(iL));
                    meaByLegacy(1,localCumCard+iL) = alongIndicesI(iL);
                    
                    
                    if(isnan(sum(hNewLegacy(1,localCumCard+iL))))
                        error('why nans in hNew ii-loop');
                    end
                    
                end
                localCumCard = localCumCard + length(downIndicesI);
                if(isnan(sum(hNewLegacy(1:localCumCard))))
                    error('why nans in hNewLegacy 1');
                end
                if(ismember(0,hNewLegacy))
                    error('Dont like 0 in hNewLegacy either');
                end
            end
            newBornTracksToActivate = indicesOfNewbornTracks(setdiff(gatedInC,meaByLegacy));
            hNew = [hNewLegacy,newBornTracksToActivate];
            
            %sh = size(hNew,2)
            
            %        error('whats up with the cardinalities');
            
            
            hyposLocal(pointerHL+1:pointerHL+size(hNew,2)) = hNew;
            hyposCardLocal(pointerCL+1) = size(hNew,2);
            probLogLocal(pointerCL+1) = pq(jj).score;
            
            pointerHL = pointerHL + size(hNew,2);
            pointerCL = pointerCL + 1;
            
            if(isnan(sum(hNew)))
                error('why nans in hNew');
            end
            
            
            
        end
        
        hyposLocal = hyposLocal(1:pointerHL);
        hyposCardLocal = hyposCardLocal(1:pointerCL);
        probLogLocal = probLogLocal(1:pointerCL);
        
        if(ismember(0,hyposLocal))
            error('why is zero member of hyposLocal');
        end
        
        if(isnan(sum(hyposLocal)))
            error('why nans in hyposLocal');
        end
        
        %error('actually start revising already here');
        
        
        % Test for suitable worked example
        
        test1 = length(pq) > 7 && length(pq) < 11;
        test2 = didAS;
        test3 = nCL == 2;
        test4 = any(hyposCardLocal > 2) && max(hyposCardLocal) <= 4;
        test5 = (lpb > -5 || lpb > uub-10e-7 ) && pq(1).score > 1;
        test6 = didExpansion && didExpansionStop;
        test7 = didLS && didLSStop;
        
        
    else  % Since I had no tracks in the best combo I construct pq(1) without solving assignment problem
        
        hyposLocal = zeros(1,0);
        hyposCardLocal = zeros(1,0);
        probLogLocal = zeros(1,0);
        
    end
    
 
    
else % I don't have any surviving hypothesis in this cluster.
    
    hyposLocal = zeros(1,0);
    hyposCardLocal = zeros(1,0);
    probLogLocal = zeros(1,0);
end


%error('kk');
% error('stop after entire while iteration');
%     if(k==2)
%        error('kkkkg');
%     end

