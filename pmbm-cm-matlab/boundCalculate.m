function pqI = boundCalculate(pqI,bestLocal)

% Function to further tighten gap between bestLocal-bound and global score for hypotheses in cluster merging of PMBM
% Written by Edmund Brekke 7th-8th October 2021
% To be revised 18th October 2021++: In addition to
% contention-gap-reduction, it should also deal with
% swith-potential-reduction.

% -------------------------------------------------------------------------
% Calculate baseline bound
% -------------------------------------------------------------------------

if(pqI.switchOrExpand.switch && ~pqI.switchOrExpand.expand)
    
    
    nCL = length(bestLocal);
    if(pqI.bound ~= 0)
        error('Bound calculation should be given a pq struct with zero bound');
    end
    
    for ii=1:nCL
        pqI.bound = pqI.bound + bestLocal(ii).scores(pqI.clustercontrib(ii).parentInBestList);
    end
    
    %baselinebound = pqI.bound
    
    % -------------------------------------------------------------------------
    % Contention gap reductions for each contention measurement
    % -------------------------------------------------------------------------
    
    % Main principle: What are the rewards of these measurements in bestLocal entries, and what are the next best rewards for the tracks claiming these?
    
    
    
    %&error('check that hypo 1 gets bound above 10');
    
    % I should be able to use sum of gapReductions after this processing?
    % Can I skip switch-potential-reductions if gapReduction has no zeros and is non-empty?
    
    %bestBoundReduction = 0;
    
    
    % -------------------------------------------------------------------------
    % Switch potential reductions
    % -------------------------------------------------------------------------
    
    % New loop over clusters
    
    boundReductions = Inf*ones(1,nCL);
    for jj=1:nCL
        if(pqI.clustercontrib(jj).switchable)
            boundReductions(jj) = pqI.bestCaseLoss(jj,pqI.clustercontrib(jj).parentInBestList) - pqI.bestCaseLoss(jj,pqI.clustercontrib(jj).parentInBestList+1);
            if(isinf(boundReductions(jj)))
                pqI.clustercontrib(jj).switchable = false;
            end
            
        end
    end
    [bestBoundReduction,bestSwitchCandidate] = min(boundReductions);
    
    
    if(any(vertcat(pqI.clustercontrib.switchable)) && ~isinf(bestBoundReduction))
        reducedBound = 0;
        for jj=1:nCL
            if(jj == bestSwitchCandidate)
                reducedBound = reducedBound + pqI.bestCaseLoss(jj,pqI.clustercontrib(jj).parentInBestList+1);
            else
                reducedBound = reducedBound + pqI.bestCaseLoss(jj,pqI.clustercontrib(jj).parentInBestList);
            end
        end
    else
        reducedBound = -Inf;
        
    end
    
    
    
%     % Old loop over clusters
%     
%     bestBoundReduction = Inf;
%     for jj=1:nCL
%         if(pqI.clustercontrib(jj).switchable)
%             jj
%             x = pqI.bestCaseLoss(jj,pqI.clustercontrib(jj).parentInBestList) - pqI.bestCaseLoss(jj,pqI.clustercontrib(jj).parentInBestList+1)
%             bestBoundReduction = min(bestBoundReduction,x);
%         end
%     end
    if(bestBoundReduction < 0 )
        error('I thought I formulated this so that bestBoundReduction also should be non-negative');
    end
    
    %bestBoundReduction
    
    % Just rewrite the whole switch potential reduction
    
    pqI.bound = max(reducedBound,pqI.score);
    
    
    
%     % Switch potential reductions could possibly bring bound below score, so prevent that from happening.
%     
%     scoreGap = pqI.bound - pqI.score;
%     gapRed = min(bestBoundReduction,scoreGap);
%     
%     
%     if(pqI.labelHypo ==2)
%        error('check L2 again'); 
%     end
%     
%     
%     if(pqI.bound - gapRed < pqI.score-100*eps)
%         error('Contention-reduced bound should never be lower than the score');
%     end
%     %pqI.bound = pqI.bound - gapRed;
%     pqI.bound = pqI.bound - gapRed+100*eps;
    

    
elseif(pqI.switchOrExpand.expand && ~pqI.switchOrExpand.switch)
    pqI.bound = pqI.score;
else
    
    error('Must be either switch or expand');
end






