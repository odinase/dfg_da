function [pqO,labelHypo] = pqAppendSwitch(pqI,parentInTree,bestLocal,toBeSwitched,nextParents,rewardMatrixCombined,optReward,tracksAllSwitch,tracksBInds,meaIndAllSwitch,clusterPerTrack,expandabilityTest,personToItem,labelHypo,upperBoundsOfScores)

% Function to make new priority queue entry after switch
% Written by Edmund Brekke during October 2021
% @pqI: Parent entry in priority queue
% @parentInTree: Where in priority queue parent entry is (previously assumed to always be the UUB holder)
% @bestLocal: Struct from parent function containing the best local hypothesis for EACH parent hypothesis.
% @toBeSwitched: Contains numbers of clusters where I plan to switch parent hypotheses.
% @nextParents: The parent hypothesis that I'm switching to for each cluster
% @bestCaseLossCopy: Outer Murty reward matrix used to chose parents in pqO
% @rewardMatrixCombined: Inner Murty reward matrix that was used to construct pqO
% @optReward: Reward value of solution of inner assignment problem.
% @tracksAllSwitch: All the tracks involved in the switching problem
% @tracksBInds: B-level indices of these tracks in their respective clusters
% @meaIndAllSwitch: All measurements involved in the switching problem
% @clusterPerTrack: The cluster membership of each track
% @expandabilityTest: Column vector which tells me which of the tracks in tracksAllSwitch that are expandable.
% @personToItem: The solution to inner assignment problem as row vector consisting of measurement indices
% @labelHypo: Counter for global hypothesis labels

nCL = length(bestLocal);
meaEO = meaIndAllSwitch(personToItem);


bestCaseLossCopy = pqI.bestCaseLoss;

% Function that makes the new pq-element after switch operation

newcontrib = struct('tracks',{},'parentInBestList',{},'parentPriorScore',{},...
    'expandable',{},'switchable',{},'tracksLocalInSuper',{},'meaIndOptLocalInSuper',{},'posteriorScore',{});

labelHypo = labelHypo + 1;


for jj=1:length(toBeSwitched)
    jSwitch = toBeSwitched(jj);
    newcontrib(jSwitch).parentPriorScore = bestLocal(jSwitch).clustercontrib(nextParents(jj)).parentPriorScore;
    newcontrib(jSwitch).parentInBestList = nextParents(jj);
    newcontrib(jSwitch).switchable = any(~isinf(bestCaseLossCopy(jSwitch,nextParents(jj)+1:end))); % Notice that vector switchabilityPossible is no more used
end
for jj=setdiff(1:nCL,toBeSwitched)
    newcontrib(jj).switchable =pqI.clustercontrib(jj).switchable;
    newcontrib(jj).parentInBestList =pqI.clustercontrib(jj).parentInBestList;
    newcontrib(jj).parentPriorScore = pqI.clustercontrib(jj).parentPriorScore;
    if(isempty(pqI.clustercontrib(jj).parentPriorScore))
        error('why empty parentPriorScore in clustercontrib');
    end
end
if(~isempty(optReward))
    score = optReward;
else
    score = 0;
end

for jj=1:nCL
    newcontrib(jj).tracksLocalInSuper = find(clusterPerTrack == jj);
    newcontrib(jj).tracks = tracksAllSwitch(clusterPerTrack == jj);
    newcontrib(jj).expandable = expandabilityTest(newcontrib(jj).tracksLocalInSuper)';
    score = score + newcontrib(jj).parentPriorScore;
    if(~isempty(newcontrib(jj).tracksLocalInSuper))
        [~,b] = ismember(meaEO(clusterPerTrack==jj),meaIndAllSwitch);
        newcontrib(jj).meaIndOptLocalInSuper = b;
        rewardSumsJ = sum(rewardMatrixCombined(m2v([newcontrib(jj).tracksLocalInSuper;newcontrib(jj).meaIndOptLocalInSuper],size(rewardMatrixCombined))));
        contribScoreJ = rewardSumsJ + newcontrib(jj).parentPriorScore;
        newcontrib(jj).posteriorScore = contribScoreJ;
    else
        newcontrib(jj).posteriorScore = newcontrib(jj).parentPriorScore;
    end
end
switchOrExpand = struct('switch',~isempty(toBeSwitched),'expand',false,'clusterBase',toBeSwitched,'trackBase',[]);

pqO = struct('clustercontrib',newcontrib,'score',score,'rewardMatrix',rewardMatrixCombined,'tracksEntire',tracksAllSwitch,...
    'meaEntire',meaIndAllSwitch,'meaEntireOpt',meaEO,...
    'tracksCLevel',clusterPerTrack,'tracksBLevel',tracksBInds,'customer2Item',personToItem,'parentInSearchTree',parentInTree,'childrenInSearchTree',zeros(1,0),...
    'childrenCount',0,'labelHypo',labelHypo,...
    'switchOrExpand',switchOrExpand,'bestCaseLoss',bestCaseLossCopy,'bound',0,'upperBoundsOfScores',upperBoundsOfScores,'hasBeenUUB',false);


% if(pqO.labelHypo == 3)
%    error('check score of H3'); 
% end

% Also verify the reward contribution from pqO

rewsumveri = 0;
for ii=1:length(personToItem)
   rewsumveri = rewsumveri + pqO.rewardMatrix(ii,personToItem(ii)); 
end
%rewsumveri


%--------------------------------------------------------------------------
% Revision October 7th++ 2021: How can I make the bounds tighter by taking measurement contention into account?

%gapReductions = gapReduce(pqO,bestLocal);
%gapRed= gapReduce(pqO,bestLocal);

% Revision October 7th++ 2021: ends
%--------------------------------------------------------------------------
                
if(any(~testBAndCpqI(pqO)))
    error('Something weith wrong with B and C level indices in pqAppendSwitch');
end

% for ii=1:nCL
%     pqO.bound = pqO.bound + bestLocal(ii).scores(pqO.clustercontrib(ii).parentInBestList);
% end
% pqO.bound = pqO.bound - gapRed;
pqO = boundCalculate(pqO,bestLocal);


