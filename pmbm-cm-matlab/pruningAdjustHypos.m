function [hyposNew,hyposCardNew,meaHistColNew,trackFileNew,trackFileShadowNew] = pruningAdjustHypos(hypos,hyposCard,meaHistCol,trackFile,trackFileShadow)

% Function to relabel tracks after missing tracks have been removed.

nT = size(meaHistCol,2);
missingTracks = setdiff(1:nT,hypos);

if(~isempty(missingTracks))
    
    
    hyposBegs = tCloud2BegInd(hyposCard);
    hyposEnds = tCloud2EndInd(hyposCard);
    
    claimedTrackBool = false(1,size(trackFile,2));
    nHNew = size(hyposCard,2);
    
    for iH=1:nHNew
        h = hypos(:,hyposBegs(iH):hyposEnds(iH));
        claimedTrackBool(h) = true;
    end
    tracksKeep = find(claimedTrackBool);
    newForEachOld = zeros(1,size(trackFile,2));
    newForEachOld(tracksKeep) = 1:size(tracksKeep,2);
    hyposNew = newForEachOld(hypos);
    hyposCardNew = hyposCard;
    
    trackFileNew = trackFile(:,tracksKeep);
    trackFileShadowNew = trackFileShadow(:,tracksKeep,:);
    meaHistColNew = meaHistCol(:,tracksKeep);
    
    
    
    %error('stop after all the stuff in pruningAdjustHypos');
    
else
   hyposNew = hypos;
   hyposCardNew = hyposCard;
   meaHistColNew = meaHistCol;
   trackFileNew = trackFile;
   trackFileShadowNew = trackFileShadow;
   
    
end
