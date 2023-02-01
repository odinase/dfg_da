function temp = plotBeforeDoubleMurty(target,yLim,hypos,hyposCard,clusters,clustersCard,trackFile,trackFileShadow,inCol,meaHistCol,predX,predZ,predP,phdTracks,muPPP,measurements,scenario,system,colorArr,nXB,nYB,k,iMC,dd)

% The function is to be used on predicted entities and measurements just before running Double Murty


% Calculate lims inside instead based on target and y-bound

tar = scenario(iMC,dd).targetsTrue(target).x(1:2,k);

xMinDisp = tar(1)-yLim*2;
xMaxDisp = tar(1)+yLim*2;
yMinDisp = tar(2)-yLim;
yMaxDisp = tar(2)+yLim;

lims = [xMinDisp,xMaxDisp,yMinDisp,yMaxDisp];


figA = figure;
set(figA,'position',[200,200,1220,510]);


temp = 0;
nTPHD = size(phdTracks,2);

% xMinDisp = lims(1);
% xMaxDisp = lims(2);
% yMinDisp = lims(3);
% yMaxDisp = lims(4);

xB= linspace(xMinDisp,xMaxDisp,nXB);
yB = linspace(yMinDisp,yMaxDisp,nYB);
phdGrid = zeros(nXB,nYB);
pMonB = zeros(2,nXB*nYB);
mapIndsB = zeros(2,nXB*nYB);
counter = 1;
for jj=1:nYB
    for ii=1:nXB
        pMonB(:,counter) = [xB(ii);yB(jj)];
        mapIndsB(:,counter) = [ii;jj];
        counter = counter + 1;
    end
end

% Visualize PHD component

phdGrid = muPPP*ones(size(phdGrid));

for ii=1:nTPHD
    muI = system.hMat*phdTracks(inCol.tarX,ii);
    coI = system.hMat*covVec2Mat(phdTracks(inCol.tarP,ii))*system.hMat';
    wI = phdTracks(inCol.exi,ii);
    vals = wI*exp(normpdfLog(pMonB,muI,coI));
    phdGrid = phdGrid + reshape(vals,[nXB,nYB]);    
end




imagesc(xB,yB,log(phdGrid'));axis xy;

%caxis([0,6e-3]);
caxis([-11.5,-5]);
hold on;


nTracks = size(meaHistCol,2);
for iT=1:nTracks
    %showTrack(iT,k,meaHistCol,trackFileShadow,scenario(iMC,dd).zList,scenario(iMC,dd).zCard,inCol,colorArr(mod(iT-1,18)+1,:));
    showTrackPred(iT,k-1,meaHistCol,trackFileShadow,predX,predP,scenario(iMC,dd).zList,scenario(iMC,dd).zCard,inCol,[1,1,1]);
    hold on;
end


for t=1:length(scenario(iMC,dd).targetsTrue)
    kTList = scenario(iMC,dd).targetsTrue(t).k;
    kI = find(scenario(iMC,dd).targetsTrue(t).k == k);
    %plot(scenario(iMC,dd).targetsTrue(t).x(1,1:kI),scenario(iMC,dd).targetsTrue(t).x(2,1:kI),'g','linewidth',2);
    plot(scenario(iMC,dd).targetsTrue(t).x(1,1:kI),scenario(iMC,dd).targetsTrue(t).x(2,1:kI),'color',colorArr(t+1,:),'linewidth',2);
    hold on;
end

plot(measurements(1,:),measurements(2,:),'ro','linewidth',2);
hold on;

hold off;
axis([xMinDisp,xMaxDisp,yMinDisp,yMaxDisp]);
%
title(['Before Double Murty at k = ',num2str(k)]);
xlabel('x [m]');
ylabel('y [m]');

write_text(predZ, figA);


