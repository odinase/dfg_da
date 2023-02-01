function [testVar,badClustersBool,hyposCardinalitiesCell] = checkForMixedCardinalities(cHypos,cHyposCard,cWNew,cCardWNew)

cBegs = tCloud2BegInd(cCardWNew);
cEnds = tCloud2EndInd(cCardWNew);
hBegs = tCloud2BegInd(cHyposCard);
hEnds = tCloud2EndInd(cHyposCard);
testVar = false;

badClustersBool = false(size(cCardWNew));
hyposCardinalitiesCell = cell(1,size(cCardWNew,2));

for iC=1:length(cBegs)
    hypoCardinalities = zeros(1,cCardWNew(iC));
    hC = cWNew(:,cBegs(iC):cEnds(iC));
    
    for iH=1:length(hC)
        tracksC = cHypos(:,hBegs(hC(iH)):hEnds(hC(iH)));
        hypoCardinalities(iH) = length(tracksC);
    end
    if(any(diff(hypoCardinalities) ~= 0))
        testVar = true;
        badClustersBool(iC) = true;
    end
    hyposCardinalitiesCell{iC} = hypoCardinalities;
end

