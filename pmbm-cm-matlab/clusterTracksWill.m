function clusters = clusterTracksWill(OmegaAssoc)

% Fuction for clustering of tracks in PMBM
% Based on earlier function for clustering of tracks in JPDA
% Written by Edmund Brekke during November/December 2017
% Revised to work with PMBM during October 2018
% @ OmegaAssoc: Validation matrix of dimension m x (n+1) 
% > clusters:   Cell array with 3 rows:
%               1st row: tracksLocal
%               2nd row: meaLocalI   
%               3rd row: OmegaAssoc(meaLocalI,tracksLocal)

%n_conf = size(OmegaAssoc,2)-1;
n_conf = size(OmegaAssoc,1);
if(n_conf < 0)
   error('cannot have negative number of tracks'); 
end


assocLocal = zeros(2,n_conf);  % Array for checking which tracks are to be clustered together
assocLocal(2,:) = assocLocal(2,:)*NaN;
for p=1:n_conf
    if(isnan(assocLocal(2,p)))
        assocLocal(2,p) = p;
        assocLocal(1,p) = 1;
    end
    for i=[1:p-1,p+1:n_conf]
        bothRow = [OmegaAssoc(p,:),OmegaAssoc(i,:)];
        if(any(sum(bothRow,1) > 1))
            if(assocLocal(2,i) > assocLocal(2,p))
                slaveIndices = find(assocLocal(2,:) == assocLocal(2,i));
                assocLocal(1,slaveIndices) = 0;
                assocLocal(2,slaveIndices) = assocLocal(2,p);
            elseif(assocLocal(2,i) < assocLocal(2,p))
                slaveIndices = find(assocLocal(2,:) == assocLocal(2,p));
                assocLocal(1,slaveIndices) = 0;
                assocLocal(2,slaveIndices) = assocLocal(2,i);
            else
                assocLocal(2,i) = assocLocal(2,p);
            end
        end
    end
end
masters = find(assocLocal(1,:));
nClusters = length(masters);
clusters = cell(3,nClusters); % 1st row: Which tracks. 2nd row: Which measurements. 3rd row: Local gating matrix
for iC=1:nClusters
    tracksLocal = find(assocLocal(2,:) == masters(iC));
    meaLocalI = find(sum(OmegaAssoc(tracksLocal,:),1) > 0)';
    clusters{1,iC} = tracksLocal;
    clusters{2,iC} = meaLocalI;
    clusters{3,iC} = OmegaAssoc(tracksLocal,meaLocalI);
end