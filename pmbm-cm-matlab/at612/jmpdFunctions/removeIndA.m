function [aIndRemove,tCloudRemove,aIndRemain,tCloudRemain] = removeIndA(aIndRemove,tCloud)

% Function to pick/move/remove A level indices based on A level indices
% Written by Edmund Brekke during December 2007
% @bRemove:         A-level indices (location in complete list)
% @tCloud:          partition lengths of original list
% >aIndRemove:      A-level indices in original list of removed elements
% >tCloudRemove:    partition lengths of removed list
% >aIndRemain:      A-level indices in original list of remaining elements
% >tCloudRemain:    partition lengths of remaining list
% $tCloud2BegInd
% $a2bc
% $tCloud2BegInd

% The ABC-hierarchy
% level A: All partitions stacked along 2nd dimension
% level B: Inside each partition
% level C: Partitions as entities

% Comment:
% Notice that this is actually more difficult than removal based on B/C
% level indices, illustrated by the extra complexity in a2bc(), which
% transforms the problem to removal based on B/C level indices.


if(max(aIndRemove) > sum(tCloud))
    error('Cannot extract higher A-level index than sum of partition lengths');
end
if(sum(aIndRemove <= 0) > 0)
    error('A-level indices must be positive');
end

% Sort aIndRemove (ideally with an algorithm good for almost sorted lists)

[aIndRemove,indSort] = sort(aIndRemove);

% Find corresponding B and C level indices

[bRemove,cRemove] = a2bc(aIndRemove,tCloud);
cRemove = cRemove(indSort);


% Make tCloudRemove (Clearly not optimal approach, make something better eventually)

cRemoveUnique = unique(cRemove);
tCloudRemove = zeros(size(cRemoveUnique));
for ii=1:length(cRemoveUnique)
    tCloudRemove(ii) = sum(cRemove == cRemoveUnique(ii));    
end

% Only determine remainder if asked to do so

if(nargout == 4)
   
    % Determine remaining A level indices

    aIndRemain = 1:sum(tCloud);
    aIndRemain(aIndRemove) = [];

    % Determine remaining partition lengths
    
    tCloudRemain = tCloud;
    for ii=1:length(cRemoveUnique)
        tCloudRemain(cRemoveUnique(ii)) = tCloudRemain(cRemoveUnique(ii)) - tCloudRemove(ii);
    end

end