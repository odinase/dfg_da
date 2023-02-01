function averages = partAverage(xCloud,tCloud)

% Modify function so that zero partitions are handled properly!

averages = zeros(size(xCloud,1),size(tCloud,2));
begInd = tCloud2BegInd(tCloud);
for t=1:size(tCloud,2)
    indices = begInd(t):(begInd(t)+tCloud(t)-1);
    if(~isempty(indices))
        averages(:,t) = mean(xCloud(:,indices),2);
    else
        averages(:,t) = NaN;
    end
end
