function [shareMeasurements,shareTracks,shareClusters] = testClustersShareMeasurements(clustersWill,clustersCardWill,hyposWill,hyposCardWill,meaHistCol)

% Function to test whether the same measurement is claimed by tracks in two different clusters
% Written by Edmund Brekke with basis in testClustersShareTracks in July 2020.
% For now, I only want to apply the test to the last measurement

shareMeasurements = zeros(1,0);
shareTracks = zeros(1,0);
shareClusters = zeros(2,0);

chBegs = tCloud2BegInd(clustersCardWill);
chEnds = tCloud2EndInd(clustersCardWill);
hBegs = tCloud2BegInd(hyposCardWill);
hEnds = tCloud2EndInd(hyposCardWill);

for iC=1:length(chBegs)
    
    
    hyposI = clustersWill(chBegs(iC):chEnds(iC));
    tracksI = hyposWill(pickIndC(hyposI,hyposCardWill));
    meaLastI = meaHistCol(end,tracksI);
    
    for jC=(iC+1):length(chBegs)
        hyposJ = clustersWill(chBegs(jC):chEnds(jC));
        
        tracksJ = hyposWill(pickIndC(hyposJ,hyposCardWill));
        meaLastJ = meaHistCol(end,tracksJ);
        

        
        
        
        meaLastI(meaLastI == 0) = NaN;
        meaLastJ(meaLastJ == 0) = NaN;
        
        C = intersect(meaLastI,meaLastJ);
        
        
        
%         if(iC == 1 && jC == 6)
%             
%             tracksI
%             tracksJ
%             meaLastI
%             meaLastJ
%             C
%             tempI = find(ismember(meaHistCol(end,tracksI),C));
%             tracksIMarked = tracksI(tempI)
%         end        
        
        if(~isempty(C))
            
            shareMeasurements = [shareMeasurements,C];
            
            
            tempI = find(ismember(meaHistCol(end,tracksI),C));
            tracksIMarked = tracksI(tempI);
            tempJ = find(ismember(meaHistCol(end,tracksJ),C));
            tracksJMarked = tracksJ(tempJ);            
            
            shareTracks = [shareTracks,tracksIMarked,tracksJMarked];
            
            
            %shareTracks = [shareTracks,intersect(tracksI,tracksJ)];
            shareClusters = [shareClusters,[iC;jC]];
            
        end
    end
end