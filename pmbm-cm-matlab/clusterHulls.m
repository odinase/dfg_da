function clusterHulls(hyposWill,hyposCardWill,clustersWill,clustersCardWill,trackFile,inCol,colorArr,k)

cBegs = tCloud2BegInd(clustersCardWill);
cEnds = tCloud2EndInd(clustersCardWill);
hBegs = tCloud2BegInd(hyposCardWill);
hEnds = tCloud2EndInd(hyposCardWill);

hold on;

for iC=1:length(clustersCardWill)
    
    hC = clustersWill(:,cBegs(iC):cEnds(iC));
    posHullC = zeros(2,0);
    for iH=1:clustersCardWill(iC)
        tH = hyposWill(hBegs(hC(iH)):hEnds(hC(iH)));
        xy = trackFile(inCol.tarX(1:2),tH);
        posHullC = [posHullC,xy];
    end
    covEmp = covarianceEmpirical(posHullC);
    if(size(posHullC,2) > 2 && det(covEmp) > 0.00001)
        ch = convhull(posHullC(1,:),posHullC(2,:));
        plot(posHullC(1,:),posHullC(2,:),'.','color',colorArr(iC,:));
        h = fill ( posHullC(1,ch), posHullC(2,ch), 'r','facealpha', 0.5 );
        set(h,'facealpha',.5,'facecolor',colorArr(iC,:));
        set(h,'edgecolor',colorArr(iC,:));
    elseif(all(diag(covEmp) < 0.001))
        plot(posHullC(1,:),posHullC(2,:),'s','color',colorArr(iC,:));
    else
        plot(posHullC(1,:),posHullC(2,:),'linewidth',2,'color',colorArr(iC,:));
    end
    %fill( posHullC(1,ch), posHullC(2,ch), 'FaceColor',[1,0,0] , 'FaceAlpha', 0.5 );
end
hold off;