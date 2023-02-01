function a = showTrackPred(trackNumber,kLast,meaHistCol,trackFileShadow,predX,predP,zList,zCard,inCol,color)

% REVISE SO THAT I CAN USE IT TO PLOT BEFORE DM
% Function to display a track with corresponding measurements and state estimates.
% Should be useful to evaluate how sensible the measurements are.

% kLast should be the time step corresponding to last row in meaHistCol


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

for kI=kBegInCol:(size(meaHistCol,1))
    %kI
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
   
    
    if(kI==6)
        
        yxi = trackFileShadow(inCol.tarX,trackNumber,kI)
    end
    
end


xHist = [xHist,predX(:,trackNumber)];
pHist = cat(3,pHist,predP(:,:,trackNumber));


%xHist(:,k) = predX(:,trackNumber);
%pHist(:,:,k) = predP(:,:,trackNumber);
xR = xHist(:,611:614)

%size(pHist)


hold on;
plot(xHist(1,:),xHist(2,:),'*-','color',color);
hold on;
plot(zHist(1,:),zHist(2,:),'ko');
el = getellipse03(xHist(1:2,end),pHist(1:2,1:2,end),1,200);
plot(el(1,:),el(2,:),'color',color);
hold off;

