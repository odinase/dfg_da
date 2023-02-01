function testBool = testTrackNumberLookup(trackNumberLookup,gainMat)

nT = size(trackNumberLookup,2);
testBool = false(1,nT);

for t=1:nT
   
    if(isnan(gainMat(trackNumberLookup(1),trackNumberLookup(2))))
        
       testBool(t) = true; 
    end
    
end