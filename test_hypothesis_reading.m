load('./data/pmbm_output_files/priorLikelihood_iMC6k1262.mat');

nT = size(trackFile,2);
m = size(measurements,2);
nC = length(clustersCard);


begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
begsH = tCloud2BegInd(hyposCard);
endsH = tCloud2EndInd(hyposCard);

for iC=1:nC
   fprintf("Cluster %d\n", iC);
   hList = clusters(begsC(iC):endsC(iC));
   nH = length(hList);
   for iH=1:nH
       tracksInH = hypos(begsH(iH):endsH(iH));
       fprintf("Tracks in hypothesis %d: ", hList(iH));
       disp(tracksInH)
   end
    
end
