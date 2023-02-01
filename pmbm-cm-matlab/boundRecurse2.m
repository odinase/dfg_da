function [bounds,localHolder] = boundRecurse2(ii,pq)

bounds = NaN*zeros(1,length(pq));
ch = pq(ii).childrenInSearchTree;
if(~isempty(ch))
    
    boundsBelowI = zeros(size(ch));
    holdersBelowI = zeros(size(ch));
    
    for jj=1:length(ch)
       [boundSJ,holderBelowI] = boundRecurse2(ch(jj),pq) ;
        boundsBelowI(jj) = boundSJ(ch(jj));
        holdersBelowI(jj) = holderBelowI;
        bounds(~isnan(boundSJ)) = boundSJ(~isnan(boundSJ));
    end
    
    [bound,ix] = max([boundsBelowI,pq(ii).score]);
    if(ix <= length(ch))
       localHolder = holdersBelowI(ix);  % HERE LOCAL UUB HOLDER IS SET
    else
        localHolder = ii;  % HERE LOCAL UUB HOLDER IS SET 
    end    
    bounds(ii) = bound;
          
    
else
    
   bound = pq(ii).bound;
   localHolder = ii;  % HERE LOCAL UUB HOLDER IS SET
   bounds(ii) = bound;

end