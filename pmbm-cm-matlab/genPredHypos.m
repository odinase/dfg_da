function [hypoPredCol,hypoPredCard,xPredCol,pPredCol,predParents,xHPredTracks,pHPredTracks,zHPredTracks]...
    = genPredHypos(bMax,hypos,hyposCard,xHCol,pHCol,xHTracks,pHTracks,zHTracks)

% Can I stop this function from generating non-Reid hypotheses?

k = size(hypos,1);
nH = length(hyposCard);

begsHypo = tCloud2BegInd(hyposCard);
endsHypo = tCloud2EndInd(hyposCard);


hypoPredCol = zeros(k+1,0);
hypoPredCard = zeros(1,0);

xPredCol = zeros(4,0);
pPredCol = zeros(10,0);
predParents = zeros(1,0);

xHPredTracks = zeros(4,0,k+1);
pHPredTracks = zeros(10,0,k+1);
zHPredTracks = zeros(2,0,k+1);

for iH=1:nH
    for b=0:bMax
        
        % Concatenate hypotheses involving newborn targets to old hypotheses
        
        hypoOldPart = hypos(:,begsHypo(iH):endsHypo(iH));
        nOld = size(hypoOldPart,2);
        hypoNewPart = zeros(1,nOld + b);
        hypoNew = [hypoOldPart,NaN*zeros(k,b); hypoNewPart];
        
        hypoPredCol = [hypoPredCol,hypoNew];
        hypoPredCard = [hypoPredCard,size(hypoNew,2)];
        
        % Corresponding concatenations for state and covariance arrays
        
        xOldPart = xHCol(:,begsHypo(iH):endsHypo(iH));
        pOldPart = pHCol(:,begsHypo(iH):endsHypo(iH));
        xPredCol = [xPredCol,xOldPart,NaN*zeros(4,b)];
        pPredCol = [pPredCol,pOldPart,NaN*zeros(10,b)];
        
        predParents = [predParents,iH];
        
        xTrackOld = xHTracks(:,begsHypo(iH):endsHypo(iH),:);
        pTrackOld = pHTracks(:,begsHypo(iH):endsHypo(iH),:);
        zTrackOld = zHTracks(:,begsHypo(iH):endsHypo(iH),:);
        
        term1 = xHPredTracks;
        term2 = cat(3,xTrackOld,NaN*ones(size(xTrackOld,1),size(xTrackOld,2),1));
        term3 = NaN*zeros(4,b,k+1);
        xHPredTracks = [term1,term2,term3];
        
        term1 = pHPredTracks;
        term2 = cat(3,pTrackOld,NaN*ones(size(pTrackOld,1),size(pTrackOld,2),1));
        term3 = NaN*zeros(10,b,k+1);
        pHPredTracks = [term1,term2,term3];
        
        term1 = zHPredTracks;
        term2 = cat(3,zTrackOld,NaN*ones(size(zTrackOld,1),size(zTrackOld,2),1));
        term3 = NaN*zeros(2,b,k+1);
        zHPredTracks = [term1,term2,term3];
        
        
    end
end