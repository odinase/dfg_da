function a = showTrack(trackNumber,kLast,meaHistCol,trackFileShadow,zList,zCard,inCol,color)

% Function to display a track with corresponding measurements and state estimates.
% Should be useful to evaluate how sensible the measurements are.

%
%figA = figure;
a = [];
zDim = size(zList,1);
xDim = length(inCol.tarX);
zHist = NaN*zeros(zDim,kLast);
xHist = NaN*zeros(xDim,kLast);
pHist = NaN*zeros(xDim,xDim,kLast);

kBegInCol = max(1,size(meaHistCol,1)-kLast+1);

zBegs = tCloud2BegInd(zCard);
zEnds = tCloud2EndInd(zCard);

for kI=kBegInCol:size(meaHistCol,1)
    k = kLast - size(meaHistCol,1) + kI;
    measurements = zList(:,zBegs(k):zEnds(k));
    jj = meaHistCol(kI,trackNumber);
    if(jj > 0 )
        zHist(:,k) = measurements(:,meaHistCol(kI,trackNumber));
    else
        zHist(:,k) = NaN*ones(zDim,1);
    end
    xHist(:,k) = trackFileShadow(inCol.tarX,trackNumber,kI);
    pHist(:,:,k) = covVec2Mat(trackFileShadow(inCol.tarP,trackNumber,kI));
    
end

hold on;
plot(xHist(1,:),xHist(2,:),'*-','color',color);
hold on;
%plot(zHist(1,:),zHist(2,:),'ko');
el = getellipse03(xHist(1:2,end),pHist(1:2,1:2,end),1,200);
plot(el(1,:),el(2,:),'color',color);
hold off;
