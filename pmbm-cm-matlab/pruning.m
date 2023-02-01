function [hypos,hyposCard,probLogHypos,xHCol,pHCol,xHTracks,pHTracks,zHTracks,designatedAfterPruning] = pruning(hypos,hyposCard,probLogHypos,xHCol,pHCol,xHTracks,pHTracks,zHTracks,designated)

% Revision feb 2018: I should inclue a mechanism for keeping the designated
% true hypothesis.
% Or perhaps a list of designated hypotheses? That could also be used in diversity pruning.
% Challenge: I also need to keep track of where they are after pruning.


[pLSorted,ixPL] = sort(probLogHypos,'descend');
[hyposIndSorted,hyposCardSorted] = pickIndC(ixPL,hyposCard);
hyposSorted = hypos(:,hyposIndSorted);
xHColSorted = xHCol(:,hyposIndSorted);
pHColSorted = pHCol(:,hyposIndSorted);
xHTracksSorted = xHTracks(:,hyposIndSorted,:);
pHTracksSorted = pHTracks(:,hyposIndSorted,:);
zHTracksSorted = zHTracks(:,hyposIndSorted,:);

probs = exp(pLSorted - pLSorted(1));
probs = probs/sum(probs);
a=1-cumsum(probs);
%toBeRemovedBool = a < 0.006 & probs < 0.006;
toBeRemovedBool = a < 0.000006 & probs < 0.000006;


% What are the new indices for designated hypotheses?




temp(ixPL) = 1:length(probLogHypos);
desigSorted = temp(designated);
toBeRemovedBool(desigSorted) = false;




% begsOrig = tCloud2BegInd(hyposCard);
% endsOrig = tCloud2EndInd(hyposCard);
% hTrueOrig = hypos(:,begsOrig(designated):endsOrig(designated));
% 
% begsS = tCloud2BegInd(hyposCardSorted);
% endsS = tCloud2EndInd(hyposCardSorted);
% hTrueS = hyposSorted(:,begsS(desigSorted):endsS(desigSorted));

% Ensure that zeroth hypothesis always is included

toBeRemovedBool(hyposCardSorted == 0) = false;
[toBeKept] = find(~toBeRemovedBool);
[hyposIndRemove,hyposCardRemove,hyposIndRemain,hyposCardRemain] = pickIndC(toBeKept,hyposCardSorted);



hypos = hyposSorted(:,hyposIndRemove);
xHCol = xHColSorted(:,hyposIndRemove);
pHCol = pHColSorted(:,hyposIndRemove);
xHTracks = xHTracksSorted(:,hyposIndRemove,:);
pHTracks = pHTracksSorted(:,hyposIndRemove,:);
zHTracks = zHTracksSorted(:,hyposIndRemove,:);

% toBeRemovedBool
% toBeKept
% hyposCardSorted
% 
% hypos
% hyposCardRemain
% hyposCardRemove
% hyposIndRemain
% hyposIndRemove

hyposCard = hyposCardRemove;
probLogHypos = pLSorted(~toBeRemovedBool);

% Change indices of designated hypotheses to conform with the removal

temp2(toBeKept) = 1:length(toBeKept);
designatedAfterPruning = temp2(desigSorted);

%begs = tCloud2BegInd(hyposCard);
%ends = tCloud2EndInd(hyposCard);
%hTrueTest = hypos(:,begs(designatedAfterPruning):ends(designatedAfterPruning))