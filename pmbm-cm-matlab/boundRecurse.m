function [bound,localHolder] = boundRecurse(ii,pq)

ch = pq(ii).childrenInSearchTree;
if(~isempty(ch))
    
    boundsBelowI = zeros(size(ch));
    holdersBelowI = zeros(size(ch));
    for jj=1:length(ch)
       [boundJ,holderBelowI] = boundRecurse(ch(jj),pq) ;
        boundsBelowI(jj) = boundJ;
        holdersBelowI(jj) = holderBelowI;
        
    end
    [bound,ix] = max([boundsBelowI,pq(ii).score]);
    if(ix <= length(ch))
       localHolder = holdersBelowI(ix); 
    else
        localHolder = ii;
    end
    
else
    
   bound = pq(ii).bound;
   localHolder = ii;
end