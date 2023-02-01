function cluster = hypo2cluster(hypo,clustersWill,clustersCard)

% Function to determine which cluster a given hypothesis belongs to

aInd = find(clustersWill == hypo);
[temp,cInd] = a2bc(aInd,clustersCard);
if(length(cInd) > 1)
   error('same hypothesis is in more than one cluster'); 
end
cluster = cInd;