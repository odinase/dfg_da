function [hyposLocal,hyposCardLocal,probLogLocal,kInvestigate,nCL,pcSize] = proptoExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoTotalMax,trackNumberLookup,nCap,mCap,k)

% Here I want to test drawing the # hypos proportional to prior probs


%nCap = 30; % Maximal number of combined prior hypotheses
%mCap = 30; % Maximal number of track-to-measurement assignments per prior hypothesis
superMinVal = -1000;
m = size(indicesOfNewbornTracks,2);
nT = size(gainMatPostC,1)-m;
lMatPostC = gainMatPostC(1:nT,1:m);


masters = find(assocLocal(2,:) == 1);  % Placeholder to be realized when separated in function
gainMatPostC(isinf(gainMatPostC)) = superMinVal;




intoThis = find(assocLocal(1,:) == masters(iC)); % Find all clusters that need to be merged with original master iC
%contentThis = clusters(pickIndC(intoThis,clustersCard)); % This would be the hypotheses of to-be-clustered clusters.
nCL = size(intoThis,2);  % Number of old clusters in local supercluster
begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
begsHypo = tCloud2BegInd(hyposCard);
endsHypo = tCloud2EndInd(hyposCard);

hInC = clusters(pickIndC(intoThis,clustersCard));
tInC = hypos(pickIndC(hInC,hyposCard));
gatedInC = find(any(~isinf(lMatPostC(tInC,:)),1));

% Make the prior-hypothesis reward matrix


hActiveList = struct('hypos',{});
firstContent = clusters(pickIndC(intoThis(1),clustersCard));
firstScores = probLogHypos(firstContent);
hActiveList(1).hypos = firstContent;

hContent = firstContent;

rewardFC = firstScores;
for jC=2:length(intoThis)
    nextContent = clusters(pickIndC(intoThis(jC),clustersCard));
    nextScores = probLogHypos(nextContent);
    n1 = size(rewardFC,1);
    n2 = size(rewardFC,2);
    rewardFC = blkdiag(rewardFC,nextScores);
    rewardFC((n1+1):end,1:n2) = -Inf;
    rewardFC(1:n1,(n2+1):end) = -Inf;
    hActiveList(jC).hypos = nextContent;
    hContent = [hContent,nextContent];
end

% Can I then run a no-nonsense Murty on rewardFC?
% Use Angel's Murty

[assignments,costs]= murty(-rewardFC,nHypoTotalMax);
bestPriorCombos = assignments'; % Column vectors are prior-combos
bestPriorComboScores = -costs;

% Then construct the prior hypotheses

pcSize = zeros(1,size(bestPriorCombos,2));
pc = struct('parents',{},'tracks',{},'score',{});
for iH=1:size(bestPriorCombos,2)
    pc(iH).parents = zeros(1,nCL);
    for ii=1:nCL
        pc(iH).parents(ii) = hContent(bestPriorCombos(ii,iH));
    end
    tInHypos = pickIndC(pc(iH).parents,hyposCard);
    pc(iH).tracks = unique(hypos(tInHypos));
    pc(iH).score = bestPriorComboScores(iH);
    pcSize(iH) = length(pc(iH).tracks);
end

% The run Murty for track-to-measurement assignment for each prior combo

pcProbs = exp(horzcat(pc.score));
pcProbs = pcProbs/sum(pcProbs);



hyposLocal = zeros(1,0);
hyposCardLocal = zeros(1,0);
probLogLocal = zeros(1,0);

for iH=1:size(bestPriorCombos,2)
   
    
    mCap = ceil(nHypoTotalMax*pcProbs(iH))
    
    % Perform ordinary Murty
    
    gainMatH = gainMatPostC(pc(iH).tracks,:);
    [assignments,costs]= murty(-gainMatH,mCap);
    %assignments(assignments > m) = 0;
    
    scores = - costs;
    goodEnoughBool = scores > superMinVal/2;
    scores = scores(goodEnoughBool);
    assignments = assignments(goodEnoughBool,:);
    
    downIndicesI = pc(iH).tracks;
    
    assignments
    % Transform these assignments to current track numbers
    
    for ii=1:size(assignments,1)
        alongIndicesI = assignments(ii,:);
        
        
        hNewLegacy = NaN*zeros(1,length(alongIndicesI));
        meaByLegacy = NaN*zeros(1,length(alongIndicesI));
        for iL=1:length(downIndicesI)
            hNewLegacy(iL) = trackNumberLookup(downIndicesI(iL),alongIndicesI(iL));
            meaByLegacy(iL) =  alongIndicesI(iL); 
        end
        newBornTracksToActivate = indicesOfNewbornTracks(setdiff(gatedInC,meaByLegacy));
        hNew = [hNewLegacy,newBornTracksToActivate];
        
        
        hyposLocal = [hyposLocal,hNew];
        hyposCardLocal = [hyposCardLocal,size(hNew,2)];
        probLogLocal = [probLogLocal,scores(ii)+pc(iH).score];
    end
    
    
%                         hNewLegacy(1,localCumCard+iL) = trackNumberLookup(downIndicesI(iL),alongIndicesI(iL));
%                     meaByLegacy(1,localCumCard+iL) = alongIndicesI(iL);
%             newBornTracksToActivate = indicesOfNewbornTracks(setdiff(gatedInC,meaByLegacy));
%             hNew = [hNewLegacy,newBornTracksToActivate];

end


error('kjhhkj');


% 
% hyposLocal = 0;
% hyposCardLocal = 0;
% probLogLocal = 0;
kInvestigate = 0;

