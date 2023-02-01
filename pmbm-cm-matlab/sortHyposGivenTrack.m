function [hyposNumbers,hyposT,hyposCardT,probsT] = sortHyposGivenTrack(hypos,hyposCard,clusters,clustersCard,probLogs,t)


% Find all hypotheses that contain given track. HyposCol contains hypothesis indices (and not track indices)

if(~isempty(intersect(t,hypos)))
    
    [clusterNumbers,hyposCol,probabilities] = track2Cluster(t,hypos,hyposCard,clusters,clustersCard,probLogs);
    
    probabilitiesCell = probLogs2Probabilities(probLogs,clusters,clustersCard);
    probabilities = zeros(1,size(clusters,2));
    cBegs = tCloud2BegInd(clustersCard);
    cEnds = tCloud2EndInd(clustersCard);
    for ii=1:size(clustersCard,2)
        probabilities(cBegs(ii):cEnds(ii)) = probabilitiesCell{ii};
    end
    
    hyposNumbersUS = zeros(1,0); % Extracted hypothesis numbers before sorting.
    for ii=1:length(hyposCol)
        hyposNumbersUS = [hyposNumbersUS,hyposCol{ii}];
    end
    [probsT,ix] = sort(probabilities(hyposNumbersUS),'descend');
    hyposNumbers = hyposNumbersUS(ix);
    
    
    %error('44');
    
    % Extract contents of selected hypotheses - pickIndC should automatically take care of sorting them as well
    
    [aRemove,tCloudRemove] = pickIndC(hyposNumbers,hyposCard);
    hyposT = hypos(aRemove);
    hyposCardT = tCloudRemove;
    
else
    
    hyposNumbers = zeros(1,0);
    hyposT = zeros(1,0);
    hyposCardT = zeros(1,0);
    probsT = zeros(1,0);
end