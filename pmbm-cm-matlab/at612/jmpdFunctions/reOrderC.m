function [xMerged,tMerged] = reOrderC(xConcat,tConcat,fwdMapping)

begIndConcat = tCloud2BegInd(tConcat);
endIndConcat = tCloud2EndInd(tConcat);

nConcat = size(tConcat,2);

tMerged = tConcat(fwdMapping);

xMerged = zeros(size(xConcat));
begIndMerged = tCloud2BegInd(tMerged);
endIndMerged = tCloud2EndInd(tMerged);

if(ismatrix(xConcat))
    for k=1:nConcat
        xMerged(:,begIndMerged(k):endIndMerged(k)) = xConcat(:,begIndConcat(fwdMapping(k)):endIndConcat(fwdMapping(k)));
    end
elseif(ndims(xConcat) == 3)
    for k=1:nConcat
        xMerged(:,begIndMerged(k):endIndMerged(k),:) = xConcat(:,begIndConcat(fwdMapping(k)):endIndConcat(fwdMapping(k)),:);
    end    
else
   error('xConcat must be 2 or 3 dimensional'); 
end