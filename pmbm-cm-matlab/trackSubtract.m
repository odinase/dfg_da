function [indices,meaHistRem] = trackSubtract(meaHistCol,meaSeq)

% Function to identify all tracks in meaHistCol that don't agree with a track in meaSeq


% First check whether the given track fits into  meaHistCol

nT = size(meaHistCol,2);
nKM = size(meaHistCol,1);
nKT = size(meaSeq,1);
remainingK = nKM-nKT;
nS = size(meaSeq,2);


if(remainingK < 0)
   error('Candidate track should not have more elements than tracks in track file'); 
end

testK = (remainingK+1):nKM;
meaHistLimited = meaHistCol(testK,:);
meaHistLimited(isnan(meaHistLimited)) = -1;


ix = zeros(1,nT);
for ii=1:nS
    
    meaSeqExt = repmat(meaSeq(:,ii),[1,nT]);
    meaSeqExt(isnan(meaSeqExt)) = -1;
    
    testVar = all(meaSeqExt == meaHistLimited ,1);
    tracks = find(testVar);
    
    ix(tracks) = 1;
end
indices = find(~ix);
meaHistRem = meaHistCol(:,indices);