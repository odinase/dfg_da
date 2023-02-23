clear all;
rewMat = -Inf*ones(5,7);
rewMat(:,1) = [3.0 ; 3.2; -3; -Inf; -Inf];
rewMat(:,2) = [-Inf; -Inf; 1.2; 3.0; -0.4];
rewMat(1,3) = -0.6;
rewMat(2,4) = -0.56;
rewMat(3,5) = -0.46;
rewMat(4,6) = -0.62;
rewMat(5,7) = -0.55;
nT = 5;
m = size(rewMat,2) - nT;

hypos = [1,2,1,3,4,5];
hyposCard = [2,2,1,1];
clusters = [1,2,3,4];
clustersCard = [2,2];
nC = length(clustersCard);

intoThis = [1,2];

begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
begsH = tCloud2BegInd(hyposCard);
endsH = tCloud2EndInd(hyposCard);

probLogHypos = log(0.5*ones(1,4));

% First I want to do brute force calcuation of normalization constant

iC = 1;
assocLocal = [1,1;1,0];
gainMatPostC = [rewMat; eye(2), -Inf*ones(2,5)];
gainMatPostC(gainMatPostC == 0) = -Inf;
nHypoTotalMax = 100;


preTN1 = ~isinf(gainMatPostC);
preTN2 = find(vec(preTN1));
trackNumberLookup = NaN*ones(size(gainMatPostC));
trackNumberLookup(preTN2) = preTN2;
k = 1;
indicesOfNewbornTracks = preTN2(end-1:end)';

gainMatPostC(isinf(gainMatPostC)) = -1000;


[hyposLocal,hyposCardLocal,probLogLocal,kInvestigate,pqLen,priorCardAve,pq] ...
    = branchAndBoundExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,...
    iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoTotalMax,trackNumberLookup,k);


%load('odinTwoCluster.mat');

tracksOfInterest = cluster2Tracks(intoThis,hypos,hyposCard,clusters,clustersCard);
meaOfInterest = find(any(gainMatPostC(tracksOfInterest,:) > -500,1));
maxN = 0;
for hh=1:length(pq)
    maxN = max(length(pq(hh).tracksEntire),maxN);
end
tnOfInterest = zeros(1,0);
tList = zeros(1,0);
mList = zeros(1,0);
normConstPost = sum(exp(vertcat(pq.score)));
probsHypos = exp(vertcat(pq.score))/normConstPost;
probsRaw = zeros(1,0);
ttpList = zeros(1,0);
for ii=1:length(tracksOfInterest)
    for jj=1:length(meaOfInterest)
        if(gainMatPostC(tracksOfInterest(ii),meaOfInterest(jj)) > -500)
            
            tnOfInterest = [tnOfInterest,trackNumberLookup(tracksOfInterest(ii),meaOfInterest(jj))];
            tList = [tList,tracksOfInterest(ii)];
            mList = [mList,meaOfInterest(jj)];
            
            
            % Identify all posterior hypotheses in which this track occurs
            
            t = tracksOfInterest(ii);
            zJ = meaOfInterest(jj);
            probSum = 0;
            
            for hh=1:length(pq)
                if(ismember(t,pq(hh).tracksEntire) && ismember(zJ,pq(hh).meaEntireOpt))
                    ix = find(pq(hh).tracksEntire == t);
                    if(pq(hh).meaEntireOpt(ix) == zJ)
                        probSum = probSum + probsHypos(hh);
                    end
                end
            end
            probsRaw = [probsRaw,probSum ];
            %ttpList = [ttpList,probSum*trackFileNew(inCol.exi,trackNumberLookup(tracksOfInterest(ii),meaOfInterest(jj)))];
        end
    end
end


hyposM = zeros(length(pq),maxN*2)*NaN;
for hh=1:length(pq)
    hyposM(hh,1:length(pq(hh).tracksEntire)) = pq(hh).tracksEntire;
    hyposM(hh,maxN+1:maxN+length(pq(hh).tracksEntire)) = pq(hh).meaEntireOpt;
end

aM = vertcat(pq.score);

% Find the Helmholtz free energy

expScores = exp(aM);
normconst = sum(expScores);
helmholtz = -log(normconst);

% Can I construct any suitable varational belief?
% Can I simply use probsRaw as a mean-field model?
% But what would then be the pmfs of the hypothesis nodes and the measurement nodes?

list = [];
for i = 1:length(pq)
    list(end+1) = pq(i).labelHypo;
end
sort(list)
nM = m;
assos = convert_hypos_to_assos(trackNumberLookup, hyposLocal, hyposCardLocal, clustersCard, pq, nT, nM);
assos'
% pMF = ones(1,length(pq));
% for t=1:length(pq)
%     tracks = pq(t).tracksEntire;
%     mea = pq(t).meaEntireOpt;
%     for ii=1:length(tracks)
%         ix = find(tList == tracks(ii) & mList == mea(ii));
%         pMF(t) = pMF(t)*probsRaw(ix);
%     end
% end
% pMF = pMF/sum(pMF);
% 
% % Make the individual track factors
% % For each track 
% % Node fields: Outcome values, marginal probability vector
% % Factor fields: Node list (max 2 nodes), joint distribution
% 
% nodes = struct('type',{},'vals',{},'marg',{}); % Type is either 'h','t' or 'm' % vals is the possible values in the outcome space, as a list
% iter = 1;
% for cc=1:length(clustersCard)
%     c = clusters(begsC(cc):endsC(cc));
%     probsC = probLogs2ProbabilitiesForHypos(c,probLogHypos,clusters,clustersCard,length(hyposCard));
%     nodes(end+1) = struct('type','h','vals',c,'marg',probsC);
% end
% for tt=1:nT
%    meas = find(gainMatPostC(tt,1:m)); 
%    measExt = [-1,0,meas];  
%    nodes(end+1) = struct('type','t','vals',measExt,'marg',ones(1,length(measExt))/length(measExt)); 
% end
% for jj=1:m
%    tras = find(gainMatPostC(1:nT,jj));
%    trasExt = [0,tras'];
%    nodes(end+1) = struct('type','m','vals',trasExt,'marg',ones(1,length(trasExt))/length(trasExt)); 
% end
% 
% factors = struct('nbh',{},'joint',{});
% for cc=1:length(clustersCard)
%     c = clusters(begsC(cc):endsC(cc));
%     
%     [probsC,aSel,b,c] = probLogs2ProbabilitiesForHypos(c,probLogHypos,clusters,clustersCard,length(hyposCard));
%     
%     % For each cluster make a new factor
%     
%     factors(end+1) = struct('nbh',cc,'joint',probsC);
%     
%     
% end
%     % Nope... the below must be changed. The loop over hypotheses must come inside of factor generation procedure, since its a cluster-track factor
%     % Or? Maybe accumulate tracks first?
%     
%     
% for cc=1:length(clustersCard)
%     c = clusters(begsC(cc):endsC(cc));    
%     nbh1 = cc;
%     tra = hypos(begsH(c(1)):endsH(c(1)));
%     for hh=2:clustersCard(cc)
%         tra = union(tra,hypos(begsH(c(hh)):endsH(c(hh))));
%     end
%     
%     for tt=1:length(tra)
%         nbh2 = nC+tra(tt);
%         nbh = [nbh1,nbh2];
%         joint = zeros(clustersCard(cc),length(nodes(nbh2).vals));
%         
%         % First two columns in joint reserved for non-existence and misdetection
%         % Then Im ready to loop through hypotheses
% 
%         for hh=1:clustersCard(cc)
%            
%             % Find all available tracks for this hypothesis
%             
%             traH = hypos(begsH(c(hh)):endsH(c(hh)));
%             
%             % If track is unavailable, the only non-existence is compatible.
%             % If track is available, then misdetection and all gated measurements are compatible.
%             
%             if(ismember(tra(tt),traH))
%                 % Track is in hypothesis
%                 joint(hh,2:end) = 1;
%             else
%                 % Track is not in hypothesis
%                 joint(hh,1) = 1;
%             end
%         end
%         factors(end+1) = struct('nbh',nbh,'joint',joint);
%     end
% end
% 
% % For each track make a new factor
% 
% for tt=1:nT
%     
%     nbh = nC + tt;
%     joint = zeros(size(nodes(tt+nC).vals));
%     joint(1) = 1;
%     joint(2) = exp(gainMatPostC(tt,m+tt));
%     otherEntries = ~isnan(trackNumberLookup(tt,1:m));
%     joint(3:end) = exp(gainMatPostC(tt,otherEntries));
%     factors(end+1) = struct('nbh',nbh,'joint',joint);
% end
% 
% 
% % For each track ot measurement connection make a new factor
% 
% for tt=1:nT
%     for jj=1:m
%         if(~isnan(trackNumberLookup(tt,jj)))
%             nbh = [nC+tt,nC + nT + jj];
%             joint = zeros(length(nodes(nbh(1).vals)),length(nodes(nbh(2).vals)));
%             error('st');
%         end
%     end
% end