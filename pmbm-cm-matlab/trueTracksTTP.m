function ttpsOfTrueTracks = trueTracksTTP(ttps,meaHistCol,hTrue,k)

% Function to calculate the ttp values of the true tracks at a given time
% Useful as a performance measure 

ttpsOfTrueTracks = zeros(1,size(hTrue,2));

% First identify column in meaHistCol (if any) that corresponds to given column in hTrue


for ii=1:size(hTrue,2)
   
    meaSeq = hTrue(1:k,ii); 
    tracksFound = findTracksForMeaSeq(meaSeq,meaHistCol);
    ttpsOfTrueTracks(ii) = sum(ttps(tracksFound));
end
