function [an,labels] = treeAncestors(treeNodes,nodeNumber)

% @treeNodes:   1st row: Parents, 2nd row: Labels
% @nodeNumber:  Chosen node whose ancestors we want
% >an:          List of all ancestors INCLUDING current node
% >labels:      Labels of ancestors, i.e., measurement numbers


% Use parent list representation of tree

hasParent = true;
an = nodeNumber;
labels = treeNodes(2,nodeNumber);
nodeCurrent = nodeNumber;
iteration = 1;
maxIter = 1000;

while(hasParent && iteration < maxIter) 
    parent = treeNodes(1,nodeCurrent);
    if(parent > 0)
        an = [parent; an];
        labels = [treeNodes(2,parent); labels];
        nodeCurrent = parent;
    else
       hasParent = false; 
    end
    iteration = iteration + 1;
end