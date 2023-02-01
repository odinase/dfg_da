function tracksFound = findTracksForMeaSeq(meaSeq,meaHistCol)

% First check whether the given track fits into  meaHistCol

nT = size(meaHistCol,2);
nKM = size(meaHistCol,1);
nKT = size(meaSeq,1);
remainingK = nKM-nKT;

if(remainingK < 0)
   
    meaSeq(1:abs(remainingK),:) = [];
    remainingK = 0;
    
    %error('Candidate track should not have more elements than tracks in track file'); 
end

testK = (remainingK+1):nKM;
meaHistLimited = meaHistCol(testK,:);
meaHistLimited(isnan(meaHistLimited)) = -1;

meaSeqExt = repmat(meaSeq,[1,nT]);

meaHistLimited(isnan(meaSeqExt)) = -1;  % This line makes findTrackAll differ from the original findTrack function

meaSeqExt(isnan(meaSeqExt)) = -1;



%testVar = all(meaSeqExt == meaHistLimited | meaSeqExt == -1,1)
testVar = all(meaSeqExt == meaHistLimited ,1);
tracksFound = find(testVar);
