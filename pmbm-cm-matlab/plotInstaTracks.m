function temp = plotInstaTracks(trackFile,inCol,plotDims,redTracks,greenTracks)

% Function to plot instantaneous track states and their covariances
% Should have access to the TTP values here for visualization

nTracks = size(trackFile,2);
blueTracks = setdiff(1:nTracks,union(redTracks,greenTracks));


states1 = trackFile(inCol.tarX(plotDims(1)),:);
states2 = trackFile(inCol.tarX(plotDims(2)),:);
covMatsEntire = covVec2Mat(trackFile(inCol.tarP,:));

fig1 = figure;
plot(states1(blueTracks),states2(blueTracks),'b*'); % I follow standard tracking convention, and not NED
hold on;
plot(states1(redTracks),states2(redTracks),'r*'); % I follow standard tracking convention, and not NED
plot(states1(greenTracks),states2(greenTracks),'g*'); % I follow standard tracking convention, and not NED

for ii=1:size(covMatsEntire,3)
    iMat = covMatsEntire(plotDims,plotDims,ii);
    el = getellipse03(trackFile(inCol.tarX(plotDims),ii),iMat,1,200);
    
    if(ismember(ii,blueTracks))
        plot(el(1,:),el(2,:),'b');
    elseif(ismember(ii,redTracks))
        plot(el(1,:),el(2,:),'r');
    elseif(ismember(ii,greenTracks))
        plot(el(1,:),el(2,:),'g');
        
    else
        error('something went wrong with blue, red and green tracks');
    end
end
hold off;
temp = 0;