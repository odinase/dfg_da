function elements = pickEltsBC(b,c,xCloud,tCloud)

% Function to pick elements based on B/C-indices
% Written by Edmund Brekke during December 2007
% @b:           Indices with partitions
% @c:           Partitions
% @xCloud:      Entire A-level element list
% @tCloud:      Lengths of all partitions
% >elements:    Picked elements
% $bc2a
% $tCloud2BegInd
% Notice that this function is just a nice wrapping of bc2a().

elements = xCloud(:,bc2a(b,c,tCloud));
