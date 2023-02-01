function trueIndex = findTrueHypothesis(hTrue,k,hyposReid,hyposCardReid)

begsHR = tCloud2BegInd(hyposCardReid);
endsHR = tCloud2EndInd(hyposCardReid);

hTrueK = hTrue(1:k,:);
nanColsBool = all(isnan(hTrueK),1);
hTrueK(:,nanColsBool) = [];
%hTrueK(:,nanColsBool) = zeros(k,0);

trueIndex = NaN;

%k
%length(hyposCardReid)

for iH=1:length(hyposCardReid)
    h = hyposReid(:,begsHR(iH):endsHR(iH));
        
    if(size(h,2) == size(hTrueK,2)) % Same number of tracks is a prerequisite for match
        test1 = isnan(h);
        test2 = isnan(hTrueK);
        nanTest = all(vec(test1 == test2));        
        if(nanTest)
            notNan = ~test1;
            hVec = h(notNan);
            hTrueVec = hTrueK(notNan);
            hyposEqual = all(hVec == hTrueVec);
            if(hyposEqual)
                trueIndex = iH;
            break;                
            end

        end
    end
end