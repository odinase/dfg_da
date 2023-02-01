figure;

trackProbs = zeros(1,size(meaHistCol,2));
for ii=1:size(meaHistCol,2)
    [clusterNumbers,hyposCol,pI] = track2Cluster(ii,hyposWill,hyposCardWill,clustersWill,clustersCardWill,probLogHypos);
    
    trackProbs(ii) = pI*trackFile(inCol.exi,ii);
end

exampleTrack = hTrue(1:5,3);
exampleTrack = [4,0,4,-1,-1]';
%exampleTrack(5) = -1;
%exampleTrack(4) = -1;
%exampleTrack(3) = -1;
[tracks,hyposCol,clusterNumbers] = findTrack(exampleTrack,meaHistCol,hyposWill,hyposCardWill,clustersWill,clustersCardWill);


tPLogs = log(trackProbs);
tPLogs = trackProbs;
tPLMin = min(tPLogs);
tPLMax = max(tPLogs);
tPLAadj = (tPLogs-tPLMin)/(tPLMax-tPLMin);

for ii=1:size(meaHistCol,2)

    pThis = tPLAadj(ii);

    
    if(ismember(ii,tracks))
        c1 = colorArr(13,1);% + (1-pThis)*(1-colorArr(13,1));
        c2 = colorArr(13,2);% + (1-pThis)*(1-colorArr(13,2));
        c3 = colorArr(13,3);% + (1-pThis)*(1-colorArr(13,3));
        c = [c1,c2,c3];        
        

        pMat = covVec2Mat(trackFile(inCol.tarP,ii));
        el = getellipse03(trackFile(inCol.tarX(1:2),ii),pMat(1:2,1:2),nSigma,200);
        plot(trackFile(inCol.tarX(1),ii),trackFile(inCol.tarX(2),ii),'*','color',c);
        hold on;
        plot(el(1,:),el(2,:),'color',c);
    else
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
    

end
axis([xMin,xMax,yMin,yMax]);
axis equal;