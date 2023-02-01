function [hyposNew,hyposCardNew,trackFile,trackFileShadow,meaHistCol,toBeKept] = pruneSelectedTracks(hypos,hyposCard,tracksToPrune,trackFile,trackFileShadow,meaHistCol,k)


% This function should work in the opposite manner of trackPruningPmbm
% I want to remove all tracks from hypos that are not in selected list (i.e., updated track file)
% And then I want to accordingly shift the track indices in hypos


nTracks = size(trackFile,2);
newTrackNumbers = zeros(1,nTracks);
counter = 1;
for ii=1:nTracks
    if(~ismember(ii,tracksToPrune))
       newTrackNumbers(ii) = counter;
       counter = counter + 1;
    end
end

%newTrackNumbers = 1:nTracks - reducings;


toBeKept = setdiff(1:nTracks,tracksToPrune);
trackFile = trackFile(:,toBeKept);
trackFileShadow = trackFileShadow(:,toBeKept,:);
meaHistCol = meaHistCol(:,toBeKept);

hyposNewA = zeros(size(hypos));

for ii=1:size(hypos,2)
    hyposNewA(ii) = newTrackNumbers(hypos(ii));
end
aIndRemove = find(hyposNewA == 0);
[aIndRemove,tCloudRemove,aIndRemain,hyposCardNew] = removeIndA(aIndRemove,hyposCard);
hyposNew = hyposNewA(aIndRemain);

%error('stop in pruneSel function');