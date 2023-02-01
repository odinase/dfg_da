function [aIndRemove,tCloudRemove,aIndRemain,tCloudRemain] = removeIndBC(bRemove,cRemove,tCloud)

% Function to pick/move/remove A-level indices based on B/C-indices
% Written by Edmund Brekke during December 2007
% @bRemove:         B-level indices (location in partition)
% @cRemove:         C-leve indices (partition number)
% @tCloud:          partition lengths of original list
% >aIndRemove:      A-level indices in original list of removed elements
% >tCloudRemove:    partition lengths of removed list
% >aIndRemove:      A-level indices in original list of remaining elements
% >tCloudRemove:    partition lengths of remaining list
% $tCloud2BegInd

% The ABC-hierarchy
% level A: All partitions stacked along 2nd dimension
% level B: Inside each partition
% level C: Partitions as entities

if(length(bRemove) ~= length(cRemove))
    error('Vectors of elements to be removed and partitions from which to remove must be of same length');
end
if(sum(bRemove <= 0) > 0 || sum(cRemove <= 0) > 0)
    error('B and C level indices must be positive');
end

% Find beginning indices of original partitions and which elements to remove

begInd = tCloud2BegInd(tCloud);
aIndRemove = begInd(cRemove) + bRemove - 1;

% Sort aIndRemove (ideally with an algorithm good for almost sorted lists)

[aIndRemove,indSort] = sort(aIndRemove);
cRemove = cRemove(indSort);

% Make tCloudRemove (Clearly not optimal approach, make something better eventually)

cRemoveUnique = unique(cRemove);
tCloudRemove = zeros(size(cRemoveUnique));
for ii=1:length(cRemoveUnique)
    tCloudRemove(ii) = sum(cRemove == cRemoveUnique(ii));    
end

% Make tCloudRemain

if(nargout == 4)

    aIndRemain = 1:sum(tCloud);
    aIndRemain(aIndRemove) = [];

    tCloudRemain = tCloud;
    for ii=1:length(cRemoveUnique)
        tCloudRemain(cRemoveUnique(ii)) = tCloudRemain(cRemoveUnique(ii)) - tCloudRemove(ii);
    end

end