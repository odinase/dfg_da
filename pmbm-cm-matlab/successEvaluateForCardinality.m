function [successOrFailure,mapCard,trueCard] = successEvaluateForCardinality(hTrue,cardsProb,k)

%@ trueCard: The true cardinality.
%@ cardsArr: Probabilities of cardinalities in posterior, starting with 0.



[temp,ix] = max(cardsProb);
mapCard = ix-1; % MAP cardinality estimate at this time / MC-run

if(~isempty(hTrue))
    hTrueCopy = hTrue;
    hTrueCopy(isnan(hTrueCopy)) = -1;
    trueCard = sum(hTrueCopy(k,:) > -1);
else
    trueCard = 0;
end

successOrFailure = mapCard == trueCard;