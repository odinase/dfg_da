function [shareTracks,shareClusters] = testClustersShareTracks(clustersWill,clustersCardWill,hyposWill,hyposCardWill)


shareTracks = zeros(1,0);
shareClusters = zeros(2,0);

chBegs = tCloud2BegInd(clustersCardWill);
chEnds = tCloud2EndInd(clustersCardWill);
hBegs = tCloud2BegInd(hyposCardWill);
hEnds = tCloud2EndInd(hyposCardWill);

for iC=1:length(chBegs)
    
    
    hyposI = clustersWill(chBegs(iC):chEnds(iC));
    tracksI = hyposWill(pickIndC(hyposI,hyposCardWill));
    for jC=(iC+1):length(chBegs)
        hyposJ = clustersWill(chBegs(jC):chEnds(jC));
        
        tracksJ = hyposWill(pickIndC(hyposJ,hyposCardWill));
        if(iC == 1 && jC==7)
            
            %error('check clusters 1 and 7');
        end
        
        
        if(~isempty(intersect(tracksI,tracksJ)))
            
            shareTracks = [shareTracks,intersect(tracksI,tracksJ)];
            shareClusters = [shareClusters,[iC;jC]];
            
        end
    end
end