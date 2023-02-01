foundInPropto = NaN*zeros(1,length(probLogLocal));
foundInBranchbound2 = NaN*zeros(1,length(probLogLocalP));

for ii=1:length(hyposCardLocal)
    
    hI = hyposLocal(begsLocalH(ii):endsLocalH(ii));
    
    % Is this hypothesis also in hyposLocal1?
    
    for jj=1:length(hyposCardLocalP)
        
        
        
        
        hP = hyposLocal2(begsLocalP(jj):endsLocalP(jj));
        
        if(jj==1)
           error('kk stop here'); 
        end
        
        
        
        if(all(ismember(hI,hP)) && isnan(foundInPropto(ii))) % All tracks in hI are also in hD
            foundInPropto(ii) = jj;
        end
        if(all(ismember(hP,hI)) && isnan(foundInBranchbound2(jj))) % All tracks in hI are also in hD
            foundInBranchbound2(jj) = ii;
        end
        
    end
end