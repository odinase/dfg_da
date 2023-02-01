function [gridX,gridY,gridVals] = mbm2phd(dimX,dimY,lims,nX,nY,clusters,clustersCard,hypos,hyposCard,probLogHypos,trackFile,inCol)


% Function to evaluate PHD grid visualization from a cluster-organized collection of MBMs
% @lims = [xMin,xMax,yMin,yMax]


xMin = lims(1);
xMax = lims(2);
yMin = lims(3);
yMax = lims(4);

gridX = linspace(xMin,xMax,nX);
gridY = linspace(yMin,yMax,nY);

%.......

pMonB = zeros(2,nX*nY);
mapIndsB = zeros(2,nX*nY);
counter = 1;
for jj=1:nY
    for ii=1:nX
        pMonB(:,counter) = [gridX(ii);gridY(jj)];
        mapIndsB(:,counter) = [ii;jj];
        counter = counter + 1;
    end
end

gridVals = zeros(nX,nY);

%.......

[trackSumProbs,trackTotalProbs] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);

for tt=1:size(trackFile,2) 
    x = trackFile(inCol.tarX([dimX,dimY]),tt);
    p = covVec2Mat(trackFile(inCol.tarP,tt));
    pRestricted = p([dimX,dimY],[dimX,dimY]);
    vals = trackTotalProbs(tt)*exp(normpdfLog(pMonB,x,pRestricted));
    gridVals = gridVals + reshape(vals,[nX,nY]);
end




% nC = size(clustersCard,2);
% cBegs = tCloud2BegInd(clustersCard);
% cEnds = tCloud2EndInd(clustersCard);
% nH = size(hyposCard,2);
% hBegs = tCloud2BegInd(hyposCard);
% hEnds = tCloud2EndInd(hyposCard);
% 
% for c=1:length(clustersCard)
%     
%     hC = clusters(cBegs(c):cEnds(c));       % Indices of the hypotheses in cluster c
%     
%     for iH=1:size(hC,2)
%         tH = hyposWill(hBegs(hC(iH)):hEnds(hC(iH)));
%         for tt=1:size(tH,2)
%            
%             x = trackFile(inCol.tarX,tH(tt));
%             p = covVec2Mat(trackFile(inCol.tarP,tH(tt)));
%             
%             e = trackFile(inCol.exi,tH(tt));
%             
%             
%             vals = wI*exp(normpdfLog(pMonB,muI,coI));
%             
%         end
%     end
%     
%     
% end

