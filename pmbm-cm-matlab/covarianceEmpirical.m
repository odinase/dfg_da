function [eMat,eta] = covarianceEmpirical(dataCloud,wCloud)



if(nargin == 1)
    eta = mean(dataCloud,2);
    sigma = zeros(size(eta,1),size(eta,1));
    for k=1:size(dataCloud,2)
        x = dataCloud(:,k);
        sigma = sigma +  (x-eta)*(x-eta)';
    end
    eMat = sigma/size(dataCloud,2);
else
    eta = dataCloud*wCloud';
    sigma = zeros(size(eta,1),size(eta,1));    
    for k=1:size(dataCloud,2)
        x = dataCloud(:,k);
        sigma = sigma +  wCloud(k)*(x-eta)*(x-eta)';
    end
    eMat = sigma;
end