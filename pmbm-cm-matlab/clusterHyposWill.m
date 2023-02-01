function [newHypos,newHyposCard,cWNew,cCardWNew,newProbLogs,assocLocal] = clusterHyposWill(lMatFull,proximities,labelShare,clustersWill,clustersCardWill,hyposWill,hyposCardWill,hyposProbLogs,k,trackFile,inCol,meaHistCol)

% New function to cluster hypotheses in PMBM
% Based on clusterTracks from JIPDA implementation
% This function should also return abc-lists of active measurements under the different clusters


begsCW = tCloud2BegInd(clustersCardWill);
endsCW = tCloud2EndInd(clustersCardWill);
%nTracks = size(clustersWill,2);

begsHCW = tCloud2BegInd(hyposCardWill);
endsHCW = tCloud2EndInd(hyposCardWill);

% New stuff

omegaAssoc = ~isinf(lMatFull);
nC = size(clustersCardWill,2);
assocLocal = zeros(2,nC);  % Array for checking which tracks are to be clustered together
assocLocal(2,:) = assocLocal(2,:)*NaN;

if(k==3)
   oat = omegaAssoc'
   lmf = lMatFull
    
end


nT = max(hyposWill);
sameZMat = zeros(nT,nT);
for p=1:nC   % Iterate through all clusters
    if(isnan(assocLocal(2,p)))
        assocLocal(2,p) = p;
        assocLocal(1,p) = 1;
    end
    for i=[1:p-1,p+1:nC]   % Iterate through the other clusters
        % What is my basic test criterion?
        % Clusters have to be merged if each contains a hypotheses which
        % each contains a track gating the same measurement
        
        hyposP = clustersWill(1,begsCW(p):endsCW(p));
        hyposI = clustersWill(1,begsCW(i):endsCW(i));
        
        if(hyposI == 0)
           error('why is hyposI 0'); 
        end
        
        tracksP = hyposWill(:,pickIndC(hyposP,hyposCardWill));
        tracksI = hyposWill(:,pickIndC(hyposI,hyposCardWill));        
        omegaP = any(omegaAssoc(tracksP,:),1);
        omegaI = any(omegaAssoc(tracksI,:),1);
        testForGatingSameZ = any(sum([omegaP;omegaI],1) > 1,2);
        
        % Revisions 7th Jan 2019: Should also include pure proximity and
        % common labels as reasons for clustering.
        
        % Proximity test
        
        testForProximity = any(vec(proximities(tracksP,tracksI)));
        testForLabelShare = any(vec(labelShare(tracksP,tracksI)));
        
        
        % common label test
        
        if(testForGatingSameZ || testForProximity || testForLabelShare)
            
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
masters = find(assocLocal(1,:));
nClusters = length(masters);   


if(k==4)
   %error('stop here'); 
end

newHypos = zeros(1,0); % To contain track numbers for each of the new hypotheses after clustering
newHyposCard = zeros(1,0);
newProbLogs = zeros(1,0);

% While I know the number of new clusters, I don't yet know the numbers of hypotheses in each cluster

cWNew = zeros(1,0);     
cCardWNew = zeros(1,nClusters);

% I should iterate only over masters!!

for iC=1:size(masters,2)
    intoThis = find(assocLocal(2,:) == masters(iC)); % Find all clusters that need to be merged with original master iC
    contentThis = clustersWill(pickIndC(intoThis,clustersCardWill)); % This would be the hypotheses of to-be-clustered clusters.  
    

    
    % Construct reward matrix to be used by clustering-purpose Murty
   
    %firstContent = pickIndC(intoThis(1),clustersCardWill);
    firstContent = clustersWill(pickIndC(intoThis(1),clustersCardWill));
    firstScores = hyposProbLogs(firstContent);
    rewardFC = firstScores; 
    for jC=2:length(intoThis)
        %nextContent = pickIndC(intoThis(jC),clustersCardWill);
        nextContent = clustersWill(pickIndC(intoThis(jC),clustersCardWill));
        nextScores = hyposProbLogs(nextContent);
        n1 = size(rewardFC,1);
        n2 = size(rewardFC,2);
        rewardFC = blkdiag(rewardFC,nextScores); % THE PRESENCE OF ZEROS IS ACTUALLY A BIG PROBLEM AND MUST BE AMENDED
        rewardFC((n1+1):end,1:n2) = -Inf;
        rewardFC(1:n1,(n2+1):end) = -Inf;
    end
    %nHMaxInClustering = 30;
    nHMaxInClustering = max(100,max(clustersCardWill(intoThis)));
 
    
    if(all(isinf(rewardFC(:))))
        error('rewardFC is Inf');
    end

    [Customer2Item,Item2Customer,Nsolutions,Rewards]=MurtyRevised(rewardFC,nHMaxInClustering);

    % Can I actually construct all these hypothese here for inspection?
    
    

    
    

    
    
    % Customer2Item here signifies choice of hypotheses from the two clusters
    
    nHyposLast = size(newHyposCard,2);
    
    newHyposContentI = zeros(length(inCol.tarX)+length(inCol.tarP),0);
    newHyposMHI = zeros(size(meaHistCol,1),0);
    newHyposExi = zeros(1,0);
    newHyposContentICard = zeros(1,0);
    for jN=1:Nsolutions
        rowOfHypos = contentThis(Customer2Item(jN,:));
        newHypos = [newHypos,hyposWill(pickIndC(rowOfHypos,hyposCardWill))];  % What I add here has to be track labels
        newHyposCard = [newHyposCard,size(pickIndC(rowOfHypos,hyposCardWill),2)]; % CAN THIS BE RIGHT? 
        newProbLogs = [newProbLogs,Rewards(jN)];
        
        newHyposContentI = [newHyposContentI,trackFile([inCol.tarX,inCol.tarP],hyposWill(pickIndC(rowOfHypos,hyposCardWill)))];
        newHyposContentICard = [newHyposContentICard, size(trackFile([inCol.tarX,inCol.tarP],hyposWill(pickIndC(rowOfHypos,hyposCardWill))),2)];

        
        newHyposMHI = [newHyposMHI,meaHistCol(:,hyposWill(pickIndC(rowOfHypos,hyposCardWill)))];
        newHyposExi = [newHyposExi,trackFile(inCol.exi,hyposWill(pickIndC(rowOfHypos,hyposCardWill)))];
%         if(k==4 && jN < 10)
%            rowOfHypos 
%            hyposWill(pickIndC(rowOfHypos,hyposCardWill))
%         end
        
        
    end
    nHyposNow = size(newHyposCard,2);
  
    begsHCI = tCloud2BegInd(newHyposContentICard);
    endsHCI = tCloud2EndInd(newHyposContentICard);
    
    

    
    if(k==4 && iC == 1)
        
        probabilitiesCell = probLogs2Probabilities(newProbLogs,1:length(newHyposCard),length(newHyposCard));
       error('check clustering for cluster 1 - in particular hypo 2'); 
    end    
    
    
    % This should give me the new hypotheses.
    % What about clusters?
    
    cWNew = [cWNew,(nHyposLast+1):nHyposNow];
    cCardWNew(iC) = nHyposNow-nHyposLast;
    
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

