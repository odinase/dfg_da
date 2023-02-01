function test = testPQLookup(pqI,trackNumberLookup)

test = false(1,length(pqI.clustercontrib));


for c=1:length(pqI.clustercontrib)
   
    
    tra = pqI.clustercontrib(c).tracks;
    me = pqI.meaEntire(pqI.clustercontrib(c).meaIndOptLocalInSuper);
    
    if(length(tra) ~= length(me))
        
        error('tra not of same length as me');
    else
        ttnl = zeros(1,length(tra));
        for ii=1:length(tra)
            
           ttnl(ii) = trackNumberLookup(tra(ii),me(ii));
        end
        if(any(isnan(ttnl)))
           test(c) = true;
           tra
           me
           
        end
        
    end
    
    
    
    
end