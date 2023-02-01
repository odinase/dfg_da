function depthForEachNode = treedepth(parentList)

depthForEachNode = zeros(size(parentList));
for ii=1:length(parentList)
    
    
    
    depth = 0;
    node = parentList(ii);
    while node~=0
        depth = depth + 1;
        node = parentList(node);
    end
    depthForEachNode(ii) = depth;
end