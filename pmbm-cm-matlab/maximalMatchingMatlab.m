function match=maximalMatchingMatlab(R)

% Function for finding a single maximal matching in bipartite graph
% Based on Hopcroft-Karp algorithm.
% Written by Edmund Brekke during November 2017
% @R:       Binary assignment matrix for the underlying assignment problem
% >match:   Array with matching nodes: length = total number of nodes. 


nT = size(R,1);
nI = size(R,2);
nN = nT+nI;
A = [zeros(nT,nT),R; R', zeros(nI,nI)];
S = sparse(A);
[rp ci]=sparse_to_csr(S);
linin = find(A(:))';
subs = v2m(linin,size(A,1));  % All the edges in the bipartite graph
[temp,ix] = sort(subs(1,:));
subs = subs(:,ix);
nE = size(subs,2);
if(nE > 0)
    match = zeros(1,nN);
    match(subs(1,1)) = subs(2,1);
    match(subs(2,1)) = subs(1,1);
    leafs = nT + (1:nI);
    stage = 1;
    while(~isempty(leafs) && stage < 100)  % While loop through stages
        
        % The following part before while is most computationally expensive
        
       
        uMatch = find(match(1:nT));
        freeU = setdiff(1:nT,uMatch);
        sq = freeU;        
        rootnodes = zeros(1,nN);
        visited = false(1,nN);
        rootnodes(freeU) = freeU;
        
        cntr = 1;
        leafs = zeros(1,0);
        ancestors = zeros(1,nN);
        layer = 1;
        layerDist = zeros(1,nN);
        layerDist(freeU) = 1;
        coda = false;        
        while(~isempty(sq) && ~(coda && (layerDist(sq(1)) > layer) ) && cntr < 100) % While loop through BFS queue 
            if(layerDist(sq(1)) > layer) % If I'm able to proceed to next layer then deactivate coda
                coda = false;
            end
            layer = layerDist(sq(1));
            
            % Pop head off queue and identify its children from bipartite graph
            
            head = sq(1);
            sq = sq(2:end);
            ch = ci(rp(head):rp(head+1)-1);
            matchHead = match(head); 
            
            % Iterate through children
            
            for k=1:length(ch)
                matchChild = match(ch(k));
                if(mod(layerDist(head),2) == 1)                             % Only consider alternating edges in BFS
                    matchTest = matchHead ~= ch(k) || matchChild ~= head;   % Odd case
                else
                    matchTest = matchHead == ch(k) && matchChild == head;   % Even case
                end
                if(layerDist(head) == 0)
                   error('layerDist should never be zero for head'); 
                end
                if(matchTest && ~visited(ch(k)))
                    ancestors(ch(k)) = head;
                    rootnodes(ch(k)) = rootnodes(head);
                    layerDist(ch(k)) = layerDist(head) + 1;

                    % If child is free and in V, then activate coda
                    
                    if(~matchChild && ch(k) > nT)
                        coda = true;
                        leafs = [leafs,ch(k)];
                        visited(ch(k)) = true;
                        break;
                    else  % Otherwise, add it to queue
                        sq = [sq,ch(k)];
                    end
                end
            end
            cntr = cntr + 1;
        end                         % End of the while loop of BFS queue
        
        % Iterate through augmenting paths given by leaf nodes.
        % Dematch accumulates edges to be removed from matching.
        % Enmatch accumulates edges to be added to matching. 
        
        dematch = false(1,nN);
        enmatch = zeros(1,nN);
        for k=1:length(leafs)
            cntr = 0;
            le = leafs(k);
            an = ancestors(leafs(k));
            while(an > 0 && cntr < 1000)
                if(match(an) == le && match(le) == an)
                    dematch(an) = true;
                    dematch(le) = true;
                elseif(match(an) ~= le && match(le) ~= an ) 
                    enmatch(an) = le;
                    enmatch(le) = an;
                end
                le = an;
                an = ancestors(an);
                cntr = cntr + 1;
            end
        end
        match(dematch) = 0;
        match(enmatch > 0) = enmatch(enmatch > 0);
        stage = stage + 1;
    end
else
   match = zeros(1,nE); 
end