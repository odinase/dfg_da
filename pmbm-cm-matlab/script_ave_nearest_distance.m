nearestDistances = zeros(8,nK);

for k=1:nK
   
    
    
    for ii=1:8
        dI = Inf*zeros(1,8);
        xI = scenario(1,2).targetsTrue(ii).x(1:2,k);
        for jj=setdiff(1:8,ii)
            xJ = scenario(1,2).targetsTrue(jj).x(1:2,k);
            dI(jj) = norm(xI-xJ);
        
        end
        nearestDistances(ii,k) = min(dI);
    end
    
    
end

