function [clustersI,hyposI] = track2ClusterAllowClusterShare(tracks,hypos,hyposCard,clusters,clustersCard)

% Function to find which cluster a track belongs to
% First find the hypotheses. Notice that a single track may belong to multiple hypotheses.
% 27th of February 2019: Extend to also return probabilities of the tracks
% Originally track2Cluster
% Revised into track2ClusterAllowClusterShare for special cases where I really want to find all clusters that a single track appears in.
% (Normally I dont want that and would rather want an error message in such cases)


%@tracks: For this function this can only be a single track
%@hypos
%@hyposCard
%@clusters
%@clustersCard
%@probLogHypos
%>clusterNumbers: Cluster membership for each track
%>hyposCol: Cell array 
%>probabilities: For each track, sum of contrib. hypothesis probababilities
%$probLogs2Probabilities
%$a2bc


if(nargin == 6 && nargout ~=3)
    error('If I have 6 inputs then there should be 3 outputs');
end
    
nT = size(tracks,2);
clusterNumbers = zeros(1,0);

%hyposCol = cell(1,nT);
% if(nargin == 6)
%     probabilities = zeros(1,nT);
%     probabilitiesCell = probLogs2Probabilities(probLogHypos,clusters,clustersCard);
%     probabilitiesH = zeros(1,length(hyposCard));
%     begsC = tCloud2BegInd(clustersCard);
%     endsC = tCloud2EndInd(clustersCard);
%     for iC=1:length(probabilitiesCell)
%         probabilitiesH(clusters(begsC(iC):endsC(iC))) = probabilitiesCell{iC};
%     end
% end






% for ii=1:nT
   hyposIA = find(hypos == tracks); 
   [temp,hyposI] = a2bc(hyposIA,hyposCard);
    
   % But hyposIA doesnt exactly give the hypothesis indices. Should it?
   
   %hyposCol{ii} = hyposIA;
   %hyposCol{ii} = hyposI;
   
   if(~isempty(hyposI))
   clustersI = zeros(size(hyposI));
   for h=1:size(hyposI,2)
       clustersIA = find(clusters == hyposI(h));
       [temp,c] = a2bc(clustersIA,clustersCard);
       clustersI(h) = c;
   end
   
%    if(length(unique(clustersI)) > 1)
%       error('Each track should only be in one clusters'); 
%    end
%    clusterNumbers(ii) = unique(clustersI);
   else
       clusterNumbers(ii) = NaN;
   end
   
%    if(nargin == 6)
%        %probabilitiesCell = probLogs2Probabilities(probLogHypos,clusters,clustersCard);
%        %probabilitiesThis = probabilitiesCell{c};
%        %probabilities(ii) = sum(probabilitiesThis(hyposI));
%        probabilities(ii) = sum(probabilitiesH(hyposI));
%    end
   
% end


