clear all;
load('scenarioMechi2PD.mat');

selectedTargets = 2; % I want to extract the measurements for these targets only.
nT = length(selectedTargets);

iMC=1;
dd = 2;

zList = scenario(iMC,dd).zList;
hTrue = scenario(iMC,dd).hTrue;
zCard = scenario(iMC,dd).zCard;

zBegs = tCloud2BegInd(zCard);
zEnds = tCloud2EndInd(zCard);

nK = length(zCard);

dimZ = size(zList,1);
dimX = 4;

meaSelArr = NaN*zeros(dimZ,nK,length(selectedTargets));

for k=1:nK
    measurements = zList(:,zBegs(k):zEnds(k));
    for t=1:length(selectedTargets)
        jMea = hTrue(k,selectedTargets(t));
        if(~isnan(jMea) && jMea > 0)
            meaSelArr(:,k,t) = measurements(:,jMea);
        end
    end
end


% Decide a time step for initialization

kInit = 7;
muPPP = 3.5368e-05;
pD = params.PDList(dd);
volPos = params.areaCircle;
lambdaFa = 0.6*params.faRate/volPos;

if(any(vec(isnan(meaSelArr(1,kInit,:)))))
    error('I shall only do initialization if I have measurents on all relevant targets');
end

xHatArr = NaN*zeros(dimX,nK,nT);
pHatArr = NaN*zeros(dimX,dimX,nK,nT);
exiArr = NaN*zeros(nK,nT);

for t=1:length(selectedTargets)
    z = meaSelArr(:,kInit,t);
    own2tar = z - params.stateFullOwn(1:2,kInit); % I can simply use z because measurements are always to be given in Cartesian world frame.
    [~,Ra] = cmAlternative(c2p(own2tar),system.rPol);
    rMat = system.rCart + Ra;
    pInit = blkdiag(rMat,params.pInitVel);
    
    
    lambdauInnerProd = pD*muPPP;
    xInit = [z;zeros(2,1)];
    
    rNumer = lambdauInnerProd;
    rDenom = lambdaFa + rNumer;
    exi = rNumer/rDenom;
    
    xHatArr(:,kInit,t) = xInit;
    pHatArr(:,:,kInit,t) = pInit;
    exiArr(kInit,t) = exi;
    
    
end

betaCompleteHist = zeros(2,nK,nT);

for k=(kInit+1):nK
    for t=1:nT
        % Predict
        
        exiOld = exiArr(k-1,t);
        exiPred = params.pS*exiOld;
        
        xOld = xHatArr(:,k-1,t);
        xPred = system.fMat*xOld;
        
        pOld = pHatArr(:,:,k-1,t);
        pPred = system.fMat*pOld*system.fMat' + system.qMat;
        
        % Update
        
        z = meaSelArr(:,k,t);
        
        if(~isnan(z(1)))
            zBar = system.hMat*xPred;
            own2tar = xPred(1:2) - params.stateFullOwn(1:2,k);
            [~,Ra] = cmAlternative(c2p(own2tar),system.rPol);
            rMat = system.rCart + Ra;
            
            sBar = system.hMat*pPred*system.hMat' + rMat;
            loglikelivals = normpdfLog(z,zBar,sBar);
            lMusicki = 1-pD + (pD/lambdaFa)*sum(exp(loglikelivals));
            
            if(isnan(sBar(1,1)))
                error('why nan in sBar');
            end
            
            
            exiPost = lMusicki*exiPred/(1-(1-lMusicki)*exiPred);
            
            betaZero = (1-pD)*lambdaFa;
            betaOther = pD*exp(loglikelivals);
            betaSum = betaZero + betaOther;
            betaZero = betaZero/betaSum;
            betaOther = betaOther/betaSum;
            
            
            if(isnan(betaZero))
                error('why nan in betaZero');
            end
            
            xEst = zeros(dimX,size(z,2)+1);
            pEst = zeros(dimX,dimX,size(z,2)+1);
            
            for jj=1:size(z,2)
                innov = z(:,jj)-zBar;
                kalmanGain = pPred*system.hMat'/sBar;
                xHat = xPred + kalmanGain*innov;
                pHat = (eye(dimX)-kalmanGain*system.hMat)*pPred;
                pHat = (pHat+pHat')/2;
                xEst(:,jj) = xHat;
                pEst(:,:,jj) = pHat;
                
                if(isnan(pHat(1,1)))
                    error('why nan in pHat');
                end
                
                
            end
            xEst(:,end) = xPred;
            pEst(:,:,end) = pPred;
            betaComplete = [betaOther,betaZero];
            
            betaCompleteHist(:,k,t) = betaComplete';
            
            [xHat,pHat] = gmReduce(xEst,pEst,betaComplete);
            
            if(k==10)
                
               %error('stop here'); 
            end
            
            
        else
            
            % What to do if I dont have a measurement?
            
            oneMinusPD = 1-pD;
            exiPost = exiPred*oneMinusPD/(1-exiPred+exiPred*oneMinusPD);
            xHat = xPred;
            pHat = pPred;
            
            %error('stop here for now');
            betaCompleteHist(2,k,t) = 1;
        end
        
        
        
        xHatArr(:,k,t) = xHat;
        pHatArr(:,:,k,t) = pHat;
        exiArr(k,t) = exiPost;
        
        %error('iqw');
    end
end





