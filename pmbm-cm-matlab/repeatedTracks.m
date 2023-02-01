function [repeatedIx] = repeatedTracks(meaHistCol)

% *Function to identify all repeated tracks
% How much can the unique function do?
% repeatedIx should be on the standard form of my association arrays: of
% size 1 x nT, where the entries give where the track is first observed in
% meaHistCol. 
% Alternatives would be cell array, nT x nT matrix or abc bookkeeping
% Include a second row for elitists as well?

nT = size(meaHistCol,2);
meaHistCol(isnan(meaHistCol)) = -1;

repeatedIx = zeros(2,nT);
for t=1:nT
    ia = ismember(meaHistCol',meaHistCol(:,t)','rows');
    q = find(ia > 0, 1, 'first');
    repeatedIx(1,ia) = q;
    repeatedIx(2,q) = 1;
end