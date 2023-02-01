function xEst = estimator3(trackFile,hypos,hyposCard,probabilities,inCol)

% Cluster-wise multi-target estimator
% Thus: hypos and hyposCard must be for a single cluster only

hBegs = tCloud2BegInd(hyposCard);
hEnds = tCloud2EndInd(hyposCard);

nH = size(hyposCard,2);
if(nH > 0)
    
    probs = probabilities; % 0-1 adjusted probabilities
    
    for ii=1:nH
        
        h = hypos(hBegs(ii):hEnds(ii));
        for tt=1:size(h,2)
            eB = trackFile(inCol.exi,h(tt));
            if(eB > 0.5)
                probs(ii) = probs(ii)*eB;
            else
                probs(ii) = probs(ii)*(1-eB);
            end
        end
    end
    [m,index_hyp]=max(probs);
    dimX = length(inCol.tarX);
    xEst = zeros(dimX,0);
    h = hypos(hBegs(index_hyp):hEnds(index_hyp));
    for tt=1:size(h,2) 
        eB = trackFile(inCol.exi,h(tt));
        if(eB > 0.5)
            xEst = [xEst,trackFile(inCol.tarX,h(tt))];
        end
    end
end