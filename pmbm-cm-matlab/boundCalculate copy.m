function pqI = boundCalculate(pqI,bestLocal)

% Function to further tighten gap between bestLocal-bound and global score for hypotheses in cluster merging of PMBM
% Written by Edmund Brekke 7th-8th October 2021
% To be revised 18th October 2021++: In addition to
% contention-gap-reduction, it should also deal with
% swith-potential-reduction.

% -------------------------------------------------------------------------
% Calculate baseline bound
% -------------------------------------------------------------------------

nCL = length(bestLocal);
if(pqI.bound ~= 0)
   error('Bound calculation should be given a pq struct with zero bound'); 
end

for ii=1:nCL
    pqI.bound = pqI.bound + bestLocal(ii).scores(pqI.clustercontrib(ii).parentInBestList);
end

% -------------------------------------------------------------------------
% Identify contended measurements
% -------------------------------------------------------------------------



contentionClustersBool = false(1,nCL);
contendedMeaActive = zeros(1,0);
for j1=1:nCL
    for j2=(j1+1):nCL
        cMea = intersect(bestLocal(j1).clustercontrib(pqI.clustercontrib(j1).parentInBestList).meaIndOpt,bestLocal(j2).clustercontrib(pqI.clustercontrib(j2).parentInBestList).meaIndOpt);
        if(~isempty(cMea))
            contentionClustersBool(j1) = true;
            contentionClustersBool(j2) = true;
            contendedMeaActive = [contendedMeaActive,cMea];
        end
    end
end
contendedMeaActive = unique(contendedMeaActive);
clustersPerContention = NaN*zeros(length(contendedMeaActive),nCL);
for jj=1:length(contendedMeaActive)
    cMea = contendedMeaActive(jj);
    counter = 1;
    for ii=1:nCL
        if(ismember(cMea,bestLocal(ii).clustercontrib(pqI.clustercontrib(ii).parentInBestList).meaIndOpt))
            clustersPerContention(jj,counter) = ii;
            counter = counter + 1;
        end
    end
end


% Rewise contention gap reduction in this if-clause

if(pqI.clustercontrib(3).parentInBestList == 3)
    gapReductions = Inf*ones(size(contendedMeaActive));
    tracksActiveForZ = zeros(1,0);
    for jj=1:length(contendedMeaActive)
        z = contendedMeaActive(jj);
        clustersCon = clustersPerContention(jj,~isnan(clustersPerContention(jj,:)));
        for ii=1:clustersCon
            for pp=1:length(bestLocal(ii).clustercontrib)
                if(ismember(z,bestLocal(ii).clustercontrib(pp).meaIndOpt))
                    tracksActiveForZ = [tracksActiveForZ,bestLocal(ii).clustercontrib(pp).tracks(bestLocal(ii).clustercontrib(pp).meaIndOpt==z)];  
                end
            end 
        end
        tracksActiveForZ = unique(tracksActiveForZ);
        for ii=1:length(tracksActiveForZ)
            t = tracksActiveForZ(ii)
%             rewardsForT = gainMatPostC(t,:); 
%             rewardsForJT = gainMatPostC(t,pqI.meaEntire == z);
%             nextBestReward = max(setdiff(rewardsForT,rewardsForJT));
%             gapReductions(jj) = min(gapReductions(jj),rewardsForJT - nextBestReward);
        end
        
        
        
    end
    
    
    
   error('check hypo 7'); 
end




% -------------------------------------------------------------------------
% Contention gap reductions for each contention measurement
% -------------------------------------------------------------------------

% Main principle: What are the rewards of these measurements in bestLocal entries, and what are the next best rewards for the tracks claiming these?

gapReductions = Inf*ones(size(contendedMeaActive));

for jj=1:length(contendedMeaActive)
    
    % Need to identify tracks claiming this measurement in bestLocal
    
    tracksRelevant = zeros(1,0);
    for ii=1:nCL
        zLocal = bestLocal(ii).clustercontrib(pqI.clustercontrib(ii).parentInBestList).meaIndOpt;
        tLocal = bestLocal(ii).clustercontrib(pqI.clustercontrib(ii).parentInBestList).tracks;
        tLocalJ = tLocal(zLocal == contendedMeaActive(jj));
        tracksRelevant = [tracksRelevant,tLocalJ];
    end
    for t=1:length(tracksRelevant)
        rewardsForT = pqI.rewardMatrix(pqI.tracksEntire == tracksRelevant(t),:); 
        rewardsForJT = pqI.rewardMatrix(pqI.tracksEntire == tracksRelevant(t),pqI.meaEntire == contendedMeaActive(jj));
        nextBestReward = max(setdiff(rewardsForT,rewardsForJT));
        gapReductions(jj) = min(gapReductions(jj),rewardsForJT - nextBestReward);
        % Positive sign means that gap can be reduced.
    end
    gapReductions(jj) = max(gapReductions(jj),0);
    
    % This will for now be followed by a check that ....
    % ... The contended measurement is in all bestLocals in its clusters
    
    clustersCon = clustersPerContention(jj,~isnan(clustersPerContention(jj,:)));
    testEveryParent = true;
    for ii=1:length(clustersCon)
        for pp=1:length(bestLocal(clustersCon(ii)).clustercontrib)
           
            if(~ismember(contendedMeaActive(jj),bestLocal(clustersCon(ii)).clustercontrib(pp).meaIndOpt))
                testEveryParent = false;
                break;
            end
        end
    end
    
    if(~testEveryParent)
        gapReductions(jj) = 0;
    end
    
end

if(pqI.clustercontrib(3).parentInBestList == 3)
   error('check hypo 7'); 
end
 
%&error('check that hypo 1 gets bound above 10');

% I should be able to use sum of gapReductions after this processing?
% Can I skip switch-potential-reductions if gapReduction has no zeros and is non-empty?

bestBoundReduction = 0;
if(isempty(gapReductions) || any(gapReductions == 0))
    
    % -------------------------------------------------------------------------
    % Switch potential reductions
    % -------------------------------------------------------------------------

    bestBoundReduction = Inf;
    for jj=1:nCL
        x = pqI.bestCaseLoss(jj,pqI.clustercontrib(jj).parentInBestList) - pqI.bestCaseLoss(jj,pqI.clustercontrib(jj).parentInBestList+1);
        bestBoundReduction = min(bestBoundReduction,x);
    end
    if(bestBoundReduction < 0 )
       error('I thought I formulated this so that bestBoundReduction also should be non-negative');
    end
    
    
    
    
    error('I got the chance to utilize switch-potential-reductions. Implement this!');
end

if(~isempty(gapReductions) && all(gapReductions == 0))
   error('Why are all gap reductions zero? Does this make sense?'); 
end

if(~isempty(gapReductions))
    
    % I have a meaningful and safe contention gap reduction.
    % No need to take switch potential reductions into account then. 
    
    gapRed = min(gapReductions(gapReductions ~= 0));
    
    %gapRed = min(bestBoundReduction,min(gapReductions(gapReductions ~= 0)));
else
    % Switch potential reductions could possibly bring bound below score, so prevent that from happening.
    
    scoreGap = pqI.bound - pqI.score;
    gapRed = min(bestBoundReduction,scoreGap); 
end
if(pqI.bound - gapRed < pqI.score-100*eps)
   error('Contention-reduced bound should never be lower than the score'); 
end
pqI.bound = pqI.bound - gapRed;







