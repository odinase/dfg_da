load('priorLikelihood612.mat');

nT = size(trackFile,2);
m = size(measurements,2);
nC = length(clustersCard);

% Plot the predicted measurements and the real measurements

fig1 = figure;
for ii=1:nT
    pos = predZ(:,ii);
    plot(pos(1),pos(2),'b.');
    hold on;
    el = getellipse03(pos,predS(:,:,ii),1,200);
    plot(el(1,:),el(2,:),'b');
end
for jj=1:m
    plot(measurements(1,jj),measurements(2,jj),'ro');
end
axis equal;
title('All predicted tracks');

% Plot predicted measurements from the nBest hypothesis in each cluster

nBest = 10;
begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
begsH = tCloud2BegInd(hyposCard);
endsH = tCloud2EndInd(hyposCard);

fig2 = figure;
for iC=1:nC
   hList = clusters(begsC(iC):endsC(iC)); 
   nH = min(length(hList),nBest);
   hBest = hList(1:nH); % Assume that probLogHypos is sorted descencingly within each cluster
   for iH=1:length(hBest)
       tracksInH = hypos(begsH(hBest(iH)):endsH(hBest(iH)));
       for ii=1:length(tracksInH)
           pos = predZ(:,tracksInH(ii));
           sCov = predS(:,:,tracksInH(ii));
           plot(pos(1),pos(2),'b.');
           hold on;
           el = getellipse03(pos,predS(:,:,tracksInH(ii)),1,200);
           plot(el(1,:),el(2,:),'b');
       end
   end
    
end
for jj=1:m
    plot(measurements(1,jj),measurements(2,jj),'ro');
end
axis equal;
title('With 10 best hypotheses per cluster');

% Solve the topmost assignment problem in C1 and related clusters

c1WithFriends = find(assocLocal(1,:) == 1);
parents = bc2a(ones(1,3),c1WithFriends,clustersCard); % One of the variable cardinality bookkeeping functions from jmpdFunctions folder
tracksInBestPriorHypo = hypos(pickIndC(parents,hyposCard)); % Another one of the variable cardinality bookkeeping functions from jmpdFunctions folder
measurementHistoriesInBestPriorHypo = meaHistCol(:,tracksInBestPriorHypo);

rewardMatrix = gainMatPostC(tracksInBestPriorHypo,:);
[personToItem,optReward,upperBoundsOfScores] = crouse2D(rewardMatrix);
for ii=1:length(personToItem)
    if(personToItem(ii) <= m)
        p1 = predZ(1:2,tracksInBestPriorHypo(ii));
        p2 = measurements(:,personToItem(ii));
        plot([p1(1),p2(1)],[p1(2),p2(2)],'g');
    else
       plot(p1(1),p1(2),'gd'); 
    end
end
hold off;




