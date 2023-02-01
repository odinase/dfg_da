clear all;
load('preExplore');


addpath('PMBM filter');
addpath('GOSPA code');
addpath('Assignment');

exploreAnalysis = struct('nHypoTotalMax',{},'nCap',{},'mCap',{},'hypoCountM',{},'hypoCountH',{},...
    'hypoCountD',{},'hypoCountDS',{},...
    'hypoCountP',{},'hypoCountPS',{},'nCL',{},'priorAveSize',{},'pqLen',{},...
    'pMassMNotH',{},'pMassMNotD',{},'pMassMNotDS',{},'pMassMNotP',{},'pMassMNotPS',{},...
    'pRatioD',{},'pRatioDS',{},'pRatioP',{},...
    'dCountRatioH',{},'dSCountRatioH',{},'pCountRatioH',{});

for ii=1:length(preExplore)
    
    if(mod(ii,100)==0)
        disp(['preExplore item number ',num2str(ii),', fraction of time completed ',num2str(ii/89491)]);
    end
    
    
    hypos = preExplore(ii).hypos;
    hyposCard = preExplore(ii).hyposCard;
    clusters = preExplore(ii).clusters;
    clustersCard = preExplore(ii).clustersCard;
    probLogHypos = preExplore(ii).probLogHypos;
    iC = preExplore(ii).iC;
    assocLocal = preExplore(ii).assocLocal;
    gainMatPostC = preExplore(ii).gainMatPostC;
    indicesOfNewbornTracks = preExplore(ii).indicesOfNewbornTracks;
    nHypoTotalMax = preExplore(ii).nHypoTotalMax;
    trackNumberLookup = preExplore(ii).trackNumberLookup;
    k     = preExplore(ii).k;

    

    
    
    [hyposLocal,hyposCardLocal,probLogLocal,kInvesti,pqLen,priorCardAve] = branchAndBoundExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoTotalMax,trackNumberLookup,k);

    
    % Find the top combinations of prior hypotheses
    % Then check quality of K-best output for
    % * Low value of K
    % * High value of K
    % * The ceiling approach
    % I find the N best prior hypotheses, where M is the default value for N
    
    nCap = 30;
    mCap = 30;
    nHypoMax = 30;
    [hyposLocalD,hyposCardLocalD,probLogLocalD,kInvestiD,nCLD,pcSizeD] = doubleMurtyExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoTotalMax,trackNumberLookup,nCap,mCap,k);
    
    [hyposLocalP,hyposCardLocalP,probLogLocalP,kInvestiP,nCLP,pcSizeP] = proptoExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoTotalMax,trackNumberLookup,nCap,mCap,k);
    
    nHypoVeryMany = 1000;
    [hyposLocalM,hyposCardLocalM,probLogLocalM,kInvestiM,pqLenM] = branchAndBoundExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoVeryMany,trackNumberLookup,k);
    
    % New exploration analysis
    
    % Make alternatives of D and P with only 150 hypotheses
    
    [probLogSortedD,ix] = sort(probLogLocalD,'descend');
    nHDS = min(length(ix),length(hyposCardLocal));
    probLogLocalDS = probLogSortedD(1:nHDS);
    hyposCardLocalDS = hyposCardLocalD(ix(1:nHDS));
    [aRemove,tCloudRemove] = pickIndC(ix(1:nHDS),hyposCardLocalD);
    hyposLocalDS = hyposLocalD(aRemove);
    
    [probLogSortedP,ix] = sort(probLogLocalP,'descend');
    nHPS = min(length(ix),length(hyposCardLocal));
    probLogLocalPS = probLogSortedP(1:nHPS);
    hyposCardLocalPS = hyposCardLocalP(ix(1:nHPS));
    [aRemove,tCloudRemove] = pickIndC(ix(1:nHPS),hyposCardLocalP);
    hyposLocalPS = hyposLocalP(aRemove);
    
    % Analyize missing probability mass versus 1000 hypos
    
    foundInBranchboundMD = NaN*zeros(1,length(probLogLocalD));
    foundInBranchboundMP = NaN*zeros(1,length(probLogLocalP));
    foundInBranchboundMDS = NaN*zeros(1,length(probLogLocalDS));
    foundInBranchboundMPS = NaN*zeros(1,length(probLogLocalPS));
    foundInBranchboundMH = NaN*zeros(1,length(probLogLocal));
    
    begsLocalH = tCloud2BegInd(hyposCardLocal);
    endsLocalH = tCloud2EndInd(hyposCardLocal);
    begsLocalM = tCloud2BegInd(hyposCardLocalM);
    endsLocalM = tCloud2EndInd(hyposCardLocalM);
    begsLocalD = tCloud2BegInd(hyposCardLocalD);
    endsLocalD = tCloud2EndInd(hyposCardLocalD);
    begsLocalP = tCloud2BegInd(hyposCardLocalP);
    endsLocalP = tCloud2EndInd(hyposCardLocalP);
    begsLocalDS = tCloud2BegInd(hyposCardLocalDS);
    endsLocalDS = tCloud2EndInd(hyposCardLocalDS);
    begsLocalPS = tCloud2BegInd(hyposCardLocalPS);
    endsLocalPS = tCloud2EndInd(hyposCardLocalPS);
    
    foundInD = NaN*zeros(1,length(probLogLocalM));
    foundInP = NaN*zeros(1,length(probLogLocalM));
    foundInDS = NaN*zeros(1,length(probLogLocalM));
    foundInPS = NaN*zeros(1,length(probLogLocalM));
    foundInH = NaN*zeros(1,length(probLogLocalM));
    
    for ii=1:length(hyposCardLocalM)
        hI = hyposLocalM(begsLocalM(ii):endsLocalM(ii));
        for jj=1:length(hyposCardLocalD)
            hD = hyposLocalD(begsLocalD(jj):endsLocalD(jj));
            if(all(ismember(hI,hD)) && isnan(foundInD(ii))) % All tracks in hI are also in hD
                foundInD(ii) = jj;
            end
        end
        for jj=1:length(hyposCardLocalDS)
            hDS = hyposLocalDS(begsLocalDS(jj):endsLocalDS(jj));
            if(all(ismember(hI,hDS)) && isnan(foundInDS(ii))) % All tracks in hI are also in hD
                foundInDS(ii) = jj;
            end
        end
        for jj=1:length(hyposCardLocalP)
            hP = hyposLocalP(begsLocalP(jj):endsLocalP(jj));
            if(all(ismember(hI,hP)) && isnan(foundInP(ii))) % All tracks in hI are also in hD
                foundInP(ii) = jj;
            end
        end
        for jj=1:length(hyposCardLocalPS)
            hPS = hyposLocalPS(begsLocalPS(jj):endsLocalPS(jj));
            if(all(ismember(hI,hPS)) && isnan(foundInPS(ii))) % All tracks in hI are also in hD
                foundInPS(ii) = jj;
            end
        end
        for jj=1:length(hyposCardLocal)
            hH = hyposLocal(begsLocalH(jj):endsLocalH(jj));
            if(all(ismember(hI,hH)) && isnan(foundInH(ii))) % All tracks in hI are also in hD
                foundInH(ii) = jj;
            end
        end
    end
    
    probabilitiesM = exp(probLogLocalM);
    probabilitiesM = probabilitiesM/sum(probabilitiesM);
    
    probsMNotH = probabilitiesM(isnan(foundInH));
    probsMNotD = probabilitiesM(isnan(foundInD));
    probsMNotDS = probabilitiesM(isnan(foundInDS));
    probsMNotP = probabilitiesM(isnan(foundInP));
    probsMNotPS = probabilitiesM(isnan(foundInPS));
    
    pMassMNotH = sum(probsMNotH);
    pMassMNotD = sum(probsMNotD);
    pMassMNotDS = sum(probsMNotDS);
    pMassMNotP = sum(probsMNotP);
    pMassMNotPS = sum(probsMNotPS);
    
    % Comparison of best score values
    
    foundInDRelH = NaN*zeros(1,length(probLogLocal));
    foundInPRelH = NaN*zeros(1,length(probLogLocal));
    foundInDSRelH = NaN*zeros(1,length(probLogLocal));
    foundInPSRelH = NaN*zeros(1,length(probLogLocal));
    
    for ii=1:length(hyposCardLocal)
        hI = hyposLocal(begsLocalH(ii):endsLocalH(ii));
        for jj=1:length(hyposCardLocalD)
            hD = hyposLocalD(begsLocalD(jj):endsLocalD(jj));
            if(all(ismember(hI,hD)) && isnan(foundInDRelH(ii))) % All tracks in hI are also in hD
                foundInDRelH(ii) = jj;
            end
        end
        for jj=1:length(hyposCardLocalDS)
            hDS = hyposLocalDS(begsLocalDS(jj):endsLocalDS(jj));
            if(all(ismember(hI,hDS)) && isnan(foundInDSRelH(ii))) % All tracks in hI are also in hD
                foundInDSRelH(ii) = jj;
            end
        end
        for jj=1:length(hyposCardLocalP)
            hP = hyposLocalP(begsLocalP(jj):endsLocalP(jj));
            if(all(ismember(hI,hP)) && isnan(foundInPRelH(ii))) % All tracks in hI are also in hD
                foundInPRelH(ii) = jj;
            end
        end
        for jj=1:length(hyposCardLocalPS)
            hPS = hyposLocalPS(begsLocalPS(jj):endsLocalPS(jj));
            if(all(ismember(hI,hPS)) && isnan(foundInPSRelH(ii))) % All tracks in hI are also in hD
                foundInPSRelH(ii) = jj;
            end
        end
    end
    
    probabilitiesH = exp(probLogLocal);
    probabilitiesH = probabilitiesH/sum(probabilitiesH);
    maxProbH = max(probabilitiesH);
    
    probsHNotD = probabilitiesH(isnan(foundInDRelH));
    probsHNotDS = probabilitiesH(isnan(foundInDSRelH));
    probsHNotP = probabilitiesH(isnan(foundInPRelH));
    probsHNotPS = probabilitiesH(isnan(foundInPSRelH));
    
    if(~isempty(probsHNotD))
        pRatioD = maxProbH/max(probsHNotD);
    else
        pRatioD = Inf;
    end
    if(~isempty(probsHNotDS))
        pRatioDS = maxProbH/max(probsHNotDS);
    else
        pRatioDS = Inf;
    end
    if(~isempty(probsHNotP))
        pRatioP = maxProbH/max(probsHNotP);
    else
        pRatioP = Inf;
    end
    
    % Ratio of how many of the hypothesis of B&B are not in the other approaches
    
    dCountRatioH = sum(isnan(foundInDRelH))/length(hyposCardLocal);
    dSCountRatioH = sum(isnan(foundInDSRelH))/length(hyposCardLocal);
    pCountRatioH = sum(isnan(foundInPRelH))/length(hyposCardLocal);
    
    exploreAnalysis(end+1) =  struct('nHypoTotalMax',nHypoTotalMax,'nCap',nCap,'mCap',mCap,...
        'hypoCountM',size(probLogLocalM,2),'hypoCountH',size(probLogLocal,2),'hypoCountD',size(probLogLocalD,2),'hypoCountDS',size(probLogLocalDS,2),...
        'hypoCountP',size(probLogLocalP,2),'hypoCountPS',size(probLogLocalPS,2),'nCL',nCLD,'priorAveSize',priorCardAve,'pqLen',pqLen,...
        'pMassMNotH',pMassMNotH,'pMassMNotD',pMassMNotD,'pMassMNotDS',pMassMNotDS,...
        'pMassMNotP',pMassMNotP,'pMassMNotPS',pMassMNotPS,...
        'pRatioD',pRatioD,'pRatioDS',pRatioDS,'pRatioP',pRatioP,...
        'dCountRatioH',dCountRatioH,'dSCountRatioH',dSCountRatioH,'pCountRatioH',pCountRatioH);
    
    
    if(size(probLogLocalP,2) < 0.6*size(probLogLocal,2) &&  size(probLogLocal,2)<80 && nCLD == 1 && priorCardAve < 2.5)
       error('why fewer hypos in P than in H'); 
    end
    
    
    if(ii==2417)
       error('what happened here'); 
    end    
    
end
