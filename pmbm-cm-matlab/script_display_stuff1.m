figure;

trackProbs = zeros(1,size(meaHistCol,2));
for ii=1:size(meaHistCol,2)
    [clusterNumbers,hyposCol,pI] = track2Cluster(ii,hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHypos);
    
    trackProbs(ii) = pI*trackFile(inCol.exi,ii);
end



tPLogs = log(trackProbs);
tPLogs = trackProbs;
tPLMin = min(tPLogs);
tPLMax = max(tPLogs);
tPLAadj = (tPLogs-tPLMin)/(tPLMax-tPLMin);

for ii=1:size(meaHistCol,2)

    pThis = tPLAadj(ii);
    c1 = colorArr(1,1) + (1-pThis)*(1-colorArr(1,1));
    c2 = colorArr(1,2) + (1-pThis)*(1-colorArr(1,2));
    c3 = colorArr(1,3) + (1-pThis)*(1-colorArr(1,3));
    c = [c1,c2,c3];
    
    
    pMat = covVec2Mat(trackFile(inCol.tarP,ii));
    el = getellipse03(trackFile(inCol.tarX(1:2),ii),pMat(1:2,1:2),nSigma,200);
    
    plot(trackFile(inCol.tarX(1),ii),trackFile(inCol.tarX(2),ii),'*','color',c);
    hold on;
    plot(el(1,:),el(2,:),'color',c);
end
axis([xMin,xMax,yMin,yMax]);
axis equal;