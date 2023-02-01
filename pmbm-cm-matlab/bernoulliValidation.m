function [meaHistColOld,trackNumberLookup,predX,predZ,predP,predS,gainMatFull,hypos,hyposCard,trackFile,trackFileShadow,meaHistCol,newTrackSubs,nTracksTenta] = bernoulliValidation(trackFile,meaHistCol,measurements,lambdauInnerProds,pD,inCol,doSemiTwoPoint,...
    predX,predZ,predP,predS,prevP,gammaGate,lambdaFa,hypos,hyposCard,clusters,clustersCard,probLogHypos,trackFileShadow,k,adoleThres)


[trackSumProbsV,trackTotalProbsV,probabilitiesCell] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);

nTracks = size(trackFile,2);
m = size(measurements,2);
meaHistColOld = meaHistCol;

gainMatFull = -Inf*ones(nTracks+m,m+nTracks);
for t=1:nTracks
    predExi = trackFile(inCol.exi,t);
    for jj=1:m
        z = measurements(:,jj);
        zBar = predZ(:,t);
        innov = z-zBar;
        sMat = predS(:,:,t);
        gateTest = innov'*(sMat\innov);
        if(gateTest < gammaGate)
            gainMatFull(t,jj) = - log(lambdaFa + pD*lambdauInnerProds(jj)) + normpdfLog(z,zBar,sMat) + log(pD) + log(predExi); % Detections of already existing targets
            % Would there be any savings from doing KF updates here as well?
        end
    end
    gainMatFull(t,m+t) = log(1-predExi + predExi*(1-pD));  % Misdetections
end
for jj=1:m
    gainMatFull(nTracks+jj,jj) = 0;
end

gainMatOriginal = gainMatFull;
if(doSemiTwoPoint)
    
    % First identify all adolescent tracks as elements in gainMatCopy2
    
    adolescents = find(sum(~isnan(meaHistCol),1) == 1);
    
    for ii=1:length(adolescents)
        
        % What I want is to first identify all non-adolescent tracks claiming the same measurement at previous time step
        % Then I want the maximal TTP-reward among these for comparison with adolescent track-measurement assignment
        
        nonAdole = find(sum(~isnan(meaHistCol),1) > 1 & meaHistCol(end,:) == meaHistCol(end,adolescents(ii)));
        nonAdoleAlphaBeta = zeros(2,size(nonAdole,2));
        
        for jj=1:size(nonAdole,2)
            nonAdoleAlphaBeta(1,jj) =  trackTotalProbsV(nonAdole(jj));
            gainMatJ = gainMatFull(jj,1:m);
            betaValues = gainMatJ(~isinf(gainMatJ));
            if(~isempty(betaValues))
                nonAdoleAlphaBeta(2,jj) = max(betaValues);
            else
                nonAdoleAlphaBeta(2,jj) = -Inf;
            end
        end
        if(~isempty(nonAdole))
            alphaBetaProdMax = max(nonAdoleAlphaBeta(1,:).*exp(nonAdoleAlphaBeta(2,:)));
            
            % Here I need a for loop for all children of adolescent number ii
            
            ch = find(gainMatFull(adolescents(ii),1:m));
            %ch = find(gainMatFull(adolescents(ii),:));
            for jj=1:length(ch)
                adoleAlphaBetaProd = trackTotalProbsV(adolescents(ii))*exp(gainMatFull(adolescents(ii),ch(jj)));
                if(alphaBetaProdMax > adoleThres*adoleAlphaBetaProd)
                    % In this case, we should disable this (adolescent) track-to-measurement assignment
                    gainMatFull(adolescents(ii),ch(jj)) = -Inf;
                end
            end
        end
    end
    
    % I should also remove all tracks that now have been made infeasible
    
    infeasible = all(isinf(gainMatFull(1:nTracks,:)),2);
    if(~isempty(infeasible))
        feasible = ~infeasible;
        
        % It seems I need a proper track pruning function here
        
        tracksToPrune = find(infeasible);
        [hypos,hyposCard,trackFile,trackFileShadow,meaHistCol,toBeKept] = pruneSelectedTracks(hypos,hyposCard,tracksToPrune,trackFile,trackFileShadow,meaHistCol,k);
        predX = predX(:,toBeKept);
        
        predP = predP(:,:,toBeKept);
        prevP = prevP(:,:,toBeKept);
        predS = predS(:,:,toBeKept);
        predZ = predZ(:,toBeKept);
        
        %meaHistFull = meaHistFull(:,toBeKept);
        
        if(max(hypos) > size(trackFile,2))
            error('hypos refers to non-existent tracks 1');
        end
        gainMatFull(infeasible,:) = [];
        gainMatFull(:,m+tracksToPrune) = [];
        nTracks = nTracks - sum(infeasible);
    end
    gainMatCopy = gainMatFull;
else
    gainMatCopy = gainMatFull;
end
if(any(all(isinf(gainMatFull(:,m+1:end)),1),2))
    error('Entire columns in misdetection part of gainMatPostC are Inf');
end

gainMatCopy((nTracks+1):end,(nTracks+1+m):end) = -Inf;
gainMatCopy = gainMatCopy';
indices = find(gainMatCopy(:)> -Inf)';
newTrackSubs = flipud(v2m(indices,size(gainMatCopy,1)));

nTracksTenta = size(newTrackSubs,2);  % Tentative number of new tracks. Some may be deleted if they fail to get hypothesis support.
trackNumberLookup = NaN*zeros(size(gainMatFull));
for ii=1:nTracksTenta
    trackNumberLookup(newTrackSubs(1,ii),newTrackSubs(2,ii)) = ii;
end