function assoc = partitionLinkage(CAND,metric)

% Generic linkage based partition function 
% Written by Edmund Brekke during July 2007
% Rewritten by Edmund Brekke during December 2007
%@ CAND:       array (B-level particle in the JMPD) to be partitioned
%@ metric:  Metric to determine when two elements interact with respect to threshold 1
%> assoc:   

% Number of (B-level) elements

k = size(CAND,2);

%Make an association list to keep order of indices
%First row: One if elitist
%Second row: Indices for corresponding elitists

assoc = zeros(2,k);
threshold = 1;

%--------------------------------------------------------------------------
%Improved associate step of the extraction---------------------------------
%--------------------------------------------------------------------------

for p=1:k
    if(assoc(2,p) == 0)
        assoc(2,p) = p; 
        assoc(1,p) = 1;
    end
    for i=[1:p-1,p+1:k]
        
        % Start machinery if element i is close enough to element p
        
        if(metric(CAND(:,i),CAND(:,p)) < threshold)
            
            % If i is unclaimed, let p claim it
            
            if(assoc(2,i) == 0)
                assoc(2,i) = assoc(2,p);
                
            % If i is claimed by parent succeeding p    
                
            elseif(assoc(2,i) > p)
                
                % Find all elements claimed by same parent as i
                
                slaveIndices = find(assoc(2,:) == assoc(2,i));
                assoc(1,slaveIndices(2:end)) = 0;
                
                % If p precedes parent transfer parentship to p
                
                if(slaveIndices(1) > p)
                    assoc(1,slaveIndices(1)) = 0;
                end
                assoc(2,slaveIndices) = assoc(2,p);
                assoc(1,i) = 0;
                
            % If i is claimed by parent preceding p    
                
            else
                
                % Find all elements claimed by same parent as p
                
                slaveIndices = find(assoc(2,:) == assoc(2,p));
                assoc(1,slaveIndices(2:end)) = 0;
                
                % If i precedes all these elements transfer parentship to i
                
                if(slaveIndices(1) > i)
                    assoc(1,slaveIndices(1)) = 0;
                end                       
                assoc(2,slaveIndices) = assoc(2,i);
                assoc(1,p) = 0;
            end 
        end      
    end
end
