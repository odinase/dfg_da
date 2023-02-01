function [trackFileNew,trackFileShadowNew,meaHistColNew,trackParents,indicesOfNewbornTracks] = trackFiltering(predX,predZ,predP,predS,newTrackSubs,measurements,system,inCol,pD,trackFile,gainMatFull,gainMatLambdau,...
    muPPP,doMechi,params,k,lambdauInnerProds,labelGen,maxLag,trackFileShadow,meaHistCol,predZLambdau,predSLambdau,predPLambdau)

m = size(measurements,2);
nTracksTenta = size(newTrackSubs,2);
nTracks = size(trackFile,2);
dimTar = size(predX,1);

indicesOfNewbornTracks = zeros(1,m);



trackFileNew = zeros(inCol.last,nTracksTenta);
trackFileShadowNew = zeros(inCol.last,nTracksTenta,maxLag);
meaHistColNew = zeros(maxLag,nTracksTenta);
trackParents = zeros(1,nTracksTenta);

% Having established the track collection, I can do KF updates for all existing tracks

for t=1:nTracksTenta
    jAss = newTrackSubs(2,t);
    
    if(newTrackSubs(1,t) <= nTracks)
        parent = newTrackSubs(1,t);
        xBar = predX(:,parent);
        zBar = predZ(:,parent);
        pBar = predP(:,:,parent);
        sMat = predS(:,:,parent);
        
        if(jAss <= m)  % A measurement is assigned to the parent track in this new track
            z = measurements(:,jAss);
            innov = z-zBar;
            kalmanGain = pBar*system.hMat'/sMat;
            xHat = xBar + kalmanGain*innov;
            pHat = (eye(dimTar)-kalmanGain*system.hMat)*pBar;
            pHat = (pHat+pHat')/2;
            jZ = jAss;
            exi = 1;
            if(det(pHat) < eps)
                error('Singular pHat');
            end
            
            
        else   % No measurement assigned to parent track. This new track constitutes a misdetection
            
            xHat = xBar;
            pHat = pBar;
            jZ = 0;
            exiOld = trackFile(inCol.exi,parent);
            %oneMinusPD = exp(gainMatFull(parent,jAss));
            oneMinusPD = 1-pD;
            exi = exiOld*oneMinusPD/(1-exiOld+exiOld*oneMinusPD);
        end
        
        
    else % Track does not have a parent. Track must therefore be initialized.
        
        z = measurements(:,jAss);
        lambdauParents = find(isinf(gainMatLambdau(:,jAss)));
        w0GM = muPPP;
        wOtherGM = exp(gainMatLambdau(lambdauParents,jAss));
        wGM = [wOtherGM;w0GM];
        wGM = wGM/sum(wGM);
        mu0GM = [z;zeros(2,1)];
        
        % Keep in mind that I want code to work both for initialization scenario, Mechi scenario and closed diver.
        % Contruction of measurement covariances should
        % have been done once and for all earlier.
        
        if(doMechi)
            own2tar = z - params.stateFullOwn(1:2,k); % I can simply use z because measurements are always to be given in Cartesian world frame.
            [~,Ra] = cmAlternative(c2p(own2tar),system.rPol);
            rMat = system.rCart + Ra;
        else
            rMat = system.rCart;
        end
        pInit = blkdiag(rMat,params.pInitVel);
        
        cov0GM = pInit;
        muOtherGM = zeros(dimTar,size(wOtherGM,1));
        covOtherGM = zeros(dimTar,dimTar,size(wOtherGM,1));
        for ii=1:size(wOtherGM,1)
            zBar = predZLambdau(:,lambdauParents(ii));
            sMat = predSLambdau(:,:,lambdauParents(ii));
            pBar = predPLambdau(:,:,lambdauParents(ii));
            innov = z-zBar;
            kalmanGain = pBar*system.hMat'/sMat;
            muOtherGM(:,ii) = xBar + kalmanGain*innov;
            covOtherGM(:,:,ii) = (eye(dimTar)-kalmanGain*system.hMat)*pBar;
        end
        muGM = [muOtherGM,mu0GM];
        covGM = cat(3,covOtherGM,cov0GM);
        [xHat,pHat] = gmReduce(muGM,covGM,wGM');
        rNumer = lambdauInnerProds(jAss);
        rDenom = params.lambdaFa + rNumer;
        exi = rNumer/rDenom;
        jZ = jAss;
        labelGen = labelGen + 1; % Label corresponds to target (or more precisely first measurement).
        parent = NaN;
        
        
    end
    
    % Store stuff in track file
    
    xStuff = zeros(inCol.last,1);
    xStuff(inCol.tarX) = xHat;
    xStuff(inCol.tarP) = covMat2Vec(pHat);
    xStuff(inCol.meaLast) = jZ;
    xStuff(inCol.cost) = gainMatFull(newTrackSubs(1),newTrackSubs(2));
    xStuff(inCol.contrib) = xStuff(inCol.cost);
    xStuff(inCol.exi) = exi;
    xStuff(inCol.visi) = 1; % SHOULD LOOK AT VISIBILITY MORE CAREFULLY LATER
    if(newTrackSubs(1,t) <= nTracks)
        xStuff(inCol.label) = trackFile(inCol.label,parent);
    else
        xStuff(inCol.label) = labelGen;
    end
    %                 trackParents = [trackParents,parent];
    %                 trackFileNew = [trackFileNew,xStuff];
    trackParents(:,t) = parent;
    trackFileNew(:,t) = xStuff;
    
    
    
    
    % I should also edit trackFileShadow here.
    % First of all, I need to settle the time indices
    % For k>maxLag, I  have to skip the first 3-slice of trackFileOld
    % For k<=maxLag,all non-NaN 3-slices of trackFileOld
    
    if(newTrackSubs(1,t) <= nTracks)
        %trackFileShadowNew = [trackFileShadowNew,cat(3,trackFileShadow(:,parent,2:end),xStuff)];
        trackFileShadowNew(:,t,:) = cat(3,trackFileShadow(:,parent,2:end),xStuff);
    else
        %trackFileShadowNew = [trackFileShadowNew,cat(3,NaN*ones(inCol.last,1,maxLag-1),xStuff)];
        trackFileShadowNew(:,t,:) = cat(3,NaN*ones(inCol.last,1,maxLag-1),xStuff);
    end
    
    % Keep track of where newborn tracks are in track file so that I can generate their hypotheses later.
    
    if(newTrackSubs(1,t) > nTracks && jZ <= m)
        indicesOfNewbornTracks(jZ) = t;
        
    end
    if(newTrackSubs(1,t) <= nTracks)
        prevPart = meaHistCol(2:end,parent);
    else
        prevPart = NaN*ones(maxLag-1,1);
    end
    hIMea = [prevPart;jZ];
    meaHistColNew(:,t) = hIMea;
    
    
    
    %error('stop here to work on meaHistFullNew');
end

% if(k==9)
%     %error('hhjk');
% end
% 


% % In future version include option to merge tracks which claim the same measurements over last maxLag scans.
% 
% proximities = zeros(nTracks,nTracks);
% labelShare = zeros(nTracks,nTracks);
% for ii=1:nTracks
%     pI = covVec2Mat(trackFile(inCol.tarP,ii));
%     labelI = trackFile(inCol.label,ii);
%     for jj=(ii+1):nTracks
%         pJ = covVec2Mat(trackFile(inCol.tarP,jj));
%         innovIJ = trackFile(inCol.tarX,ii)-trackFile(inCol.tarX,jj);
%         tTest = innovIJ'*((pI+pJ)\innovIJ);
%         if(tTest < dimTar^2)
%             proximities(ii,jj) = 1;
%         end
%         labelJ = trackFile(inCol.label,jj);
%         if(labelI == labelJ)
%             labelShare(ii,jj) = 1;
%         end
%     end
% end
% proximities = (proximities + proximities') > 0;
% labelShare = (labelShare + labelShare') > 0;

