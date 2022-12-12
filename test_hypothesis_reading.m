load('./data/pmbm_output_files/priorLikelihood_iMC7k1210.mat');

nT = size(trackFile,2);
m = size(measurements,2);
nC = length(clustersCard);


begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
begsH = tCloud2BegInd(hyposCard);
endsH = tCloud2EndInd(hyposCard);

for iC=1:nC
   fprintf("\n\nCluster %d\n", iC);
   hList = clusters(begsC(iC):endsC(iC));
   nH = length(hList);
   % Compute probs and normalize
   logprobs = probLogHypos(hList);
   lognormconst = logsumexp(logprobs);
   probs = exp(logprobs - lognormconst);
   for iH=1:nH
       tracksInH = hypos(begsH(hList(iH)):endsH(hList(iH)));
       fprintf("\nHypothesis %d:\n\tTracks: ", hList(iH));
       disp(tracksInH)
       fprintf("\tProbability: %f\n", probs(iH));
   end
end
