function [xArrNew,pArrNew,zArrNew] = kfStuff(hNew,k,xBarArr,pBarArr,sArr,pInit,measurements,dimTar,dimZ,hMat,generateXP)

xArrNew = zeros(dimTar,size(hNew,2));
pArrNew = zeros(10,size(hNew,2));
zArrNew = NaN*ones(dimZ,size(hNew,2));
for t=1:size(hNew,2)
    if(size(hNew,1) > 1 && ~isnan(hNew(k-1,t)))  % Track already exists
        xBar = xBarArr(:,t);
        pBar = pBarArr(:,:,t);
        if(hNew(k,t) > 0)       % Have detection
            sBar = sArr(:,:,t);
            z = measurements(:,hNew(k,t));
            innov = z-hMat*xBar;
            [xHat,pHatStuff] = generateXP(xBar,sBar,pBar,innov);
            pHat = covVec2Mat(pHatStuff);
            zArrNew(:,t)  = measurements(:,hNew(k,t));
        else                    % Misdetection
            xHat = xBar;
            pHat = pBar;
        end
        
    else   % Track does not exist
        z = measurements(:,hNew(k,t));
        xHat = [z;zeros(2,1)];
        pHat = pInit;
        zArrNew(:,t)  = measurements(:,hNew(k,t));
    end
    xArrNew(:,t) = xHat;
    pArrNew(:,t) = covMat2Vec(pHat);
end