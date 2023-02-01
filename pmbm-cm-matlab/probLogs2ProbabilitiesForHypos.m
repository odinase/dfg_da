function [probs,aSel,b,c] = probLogs2ProbabilitiesForHypos(hyposSelected,probLogs,clusters,clustersCard,nH)

% Function to calculate probabilities of a selected number of hypotheses


probabilitiesCell = probLogs2Probabilities(probLogs,clusters,clustersCard);

% What are the a-level indices of each hypothesis?

a = zeros(1,nH)*NaN;
a(clusters) = 1:nH;

aSel = a(hyposSelected);

[b,c] = a2bcFaster(a(hyposSelected),clustersCard);

probs = zeros(1,size(hyposSelected,2));


for ii=1:length(probs)
   
    cI = c(ii);
    bI = b(ii);
    pI = probabilitiesCell{cI};
    probs(ii) = pI(bI);
end
