function assos = convert_hypos_to_assos(trackNumberLookup, hypos, hyposCard, clustersCard, pq, nT, nM)

% For every hypothesis, find the association that was made

begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
begsH = tCloud2BegInd(hyposCard);
endsH = tCloud2EndInd(hyposCard);

nH = length(pq);
assos = zeros(nT, nH);

for iH = 1:nH
    hypo_idx = pq(iH).labelHypo;
    sH = begsH(hypo_idx);
    eH = endsH(hypo_idx);
    hh = hypos(sH:eH);
    for i = 1:length(hh)
        [tt, mm] = trackNumber2Asso(hh(i), trackNumberLookup);
        if mm > nM
            assos(tt, hypo_idx) = 0;
        else
            assos(tt, hypo_idx) = mm;
        end
    end

end

end


function [track, measurement] = trackNumber2Asso(trackNumber, trackNumberLookup)
    [track, measurement] = find(trackNumberLookup == trackNumber);
end