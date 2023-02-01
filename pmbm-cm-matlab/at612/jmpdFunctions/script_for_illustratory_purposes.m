% This script provides an illustration of how to use the functions
% * begInd2TCloud
% * endInd2TCloud
% * tCloud2BegInd
% * tCloud2EndInd

nK = 20;
zDim = 2;

% Generate measurement sets of VARIABLE CARDINALITY and store them in a
% matrix (thus AVOIDING CELL ARRAYS and all that mess).

zCollection = [];
zCollectionCardinalities = zeros(1,nK);
for k=1:nK
     nZK = poissrnd(3);
     zK = rand(zDim,nZK);     
     zCollection = [zCollection,zK];
     zCollectionCardinalities(k) = nZK;
end

% Pick out current measurement set, print cardinality and possibly do some
% filtering or whatever in a more advanced version of this program.

begIndZCol = tCloud2BegInd(zCollectionCardinalities);
endIndZCol = tCloud2EndInd(zCollectionCardinalities);
for k=1:nK
    zK = zCollection(:,begIndZCol(k):endIndZCol(k));
    nK = size(zK,2);
    disp(['Got ',num2str(nK),' measurements at time ',num2str(k)]);
end


