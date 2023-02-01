function [c,cHull,tracks,hyposCol,pAve,gridX,gridY,gridVals] = trueTrack2Cluster(trueTrackNumber,dimX,dimY,hTrue,targetsTrue,meaHistCol,hypos,hyposCard,clusters,clustersCard,probLogHypos,trackFile,inCol,nX,nY,k)


meaSeq = hTrue(:,trueTrackNumber);

% Remove any unknown-target zeros (i.e. between NaNs and integers)

firstNotNan = -1;
firstInteger = -1;
for k=1:size(meaSeq,1)
    if(~isnan(meaSeq(k)))
       firstNotNan = k;
       break;
    end
end
for k=1:size(meaSeq,1)
    if(~isnan(meaSeq(k)) && meaSeq(k) > 0)
       firstInteger = k;
       break;
    end
end

meaSeq(firstNotNan:(firstInteger-1)) = NaN;

% Still need to identify corresponding track number (if it exists)

[tracks,hyposCol,clusterNumbers] = findTrack(meaSeq,meaHistCol,hypos,hyposCard,clusters,clustersCard);

if(~isempty(tracks))
    
    c = clusterNumbers;
    [cHull,tracksInvolved] = clusterHullCalculate(dimX,dimY,c,hypos,hyposCard,clusters,clustersCard,trackFile,inCol);
    
    % Can I also say something about typical covariance for the tracks involved here?
    
    pCol = covVec2Mat(trackFile(inCol.tarP,tracksInvolved));
    pAve = mean(pCol([dimX,dimY],[dimX,dimY],:),3);
    
else
        
   xTrue = targetsTrue(trueTrackNumber).x;
   kTrue = targetsTrue(trueTrackNumber).k;
   kI = find(kTrue  == k);
   
   xNow = xTrue(:,kI(1));
   
   % How to find the cluster that either is covering xNow or is closest to xNow?
   % Among all my tracks, fine the one nearest to xnow
   
   
   
   
   nT = size(meaHistCol,2);
   mahas = zeros(1,nT);
   for t=1:nT
       xT = trackFile(inCol.tarX,t);
       pT = covVec2Mat(trackFile(inCol.tarP,t));
       mahas(t) = (xT-xNow)'*(pT\(xT-xNow));
   end
   [temp,nearest] = min(mahas); 
   meaSeqAlt = meaHistCol(:,nearest);
   [tracks,hyposCol,clusterNumbers] = findTrack(meaSeqAlt,meaHistCol,hypos,hyposCard,clusters,clustersCard);
       
    c = clusterNumbers;
    [cHull,tracksInvolved] = clusterHullCalculate(dimX,dimY,c,hypos,hyposCard,clusters,clustersCard,trackFile,inCol);
    cHull = [cHull,xNow([dimX,dimY])];
    
    
    pCol = covVec2Mat(trackFile(inCol.tarP,tracksInvolved));
    pAve = mean(pCol([dimX,dimY],[dimX,dimY],:),3);
    
    
    
   %error('Just make it an error for now if the correct track cannot be found in MBM'); 
end

% Also make MBM-PHD grid

xMin = min(cHull(1,:)) - sqrt(pAve(1,1));
xMax = max(cHull(1,:)) + sqrt(pAve(1,1));
yMin = min(cHull(2,:)) - sqrt(pAve(2,2));
yMax = max(cHull(2,:)) + sqrt(pAve(2,2));

lims = [xMin,xMax,yMin,yMax];

[gridX,gridY,gridVals] = mbm2phd(dimX,dimY,lims,nX,nY,clusters,clustersCard,hypos,hyposCard,probLogHypos,trackFile,inCol);




