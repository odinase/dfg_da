function [boolsH,boolsT] = testRepeatedMeasurements(hypos,hyposCard,meaHistCol)

% Function for testing whether a current measurement is repeated in any hypotheses

begsH = tCloud2BegInd(hyposCard);
endsH = tCloud2EndInd(hyposCard);

boolsH = false(1,size(hyposCard,2));
boolsT = false(1,size(meaHistCol,2));
for ii=1:size(hyposCard,2)
   h = hypos(begsH(ii):endsH(ii));
   z = meaHistCol(end,h);
   uniZ = unique(z);
   uniZ(uniZ==0) = [];
   counts = histc(z,uniZ);
   if(any(counts > 1))
       boolsH(ii) = true;
       if(nargout > 1)
           zIndsRep = unique(z(find(counts > 1)));
           for jj=1:size(zIndsRep)
              bInds = find(z == zIndsRep(jj));
              boolsT(h(bInds)) = true;
           end
       end
   end
end
