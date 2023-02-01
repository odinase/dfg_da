function [gainMat,pBarArr,sArr] = makeGainMatrix(fMat,qMat,hMat,rMat,xBarArr,measurements,pHatArr,gammaGate,lambdaFa,pD,muBirth,n,dimTar,dimZ)

% 180914: Revise this function to PMBM formalism

% n is the number of tracks in parent hypothesis (N_TGT)

m = size(measurements,2);
pBarArr = zeros(dimTar,dimTar,n);
sArr = zeros(dimZ,dimZ,n);

gainMat = -Inf*ones(n+m,2*m+n);
for t=1:n
    pBarArr(:,:,t) = fMat*pHatArr(:,:,t)*fMat' + qMat;
    sArr(:,:,t) = hMat*pBarArr(:,:,t)*hMat' + rMat;
    for jj=1:m
        zBar = hMat*xBarArr(:,t);
        z = measurements(:,jj);
        innov = z-zBar;
        sMat = sArr(:,:,t);
        gateTest = innov'*(sMat\innov);
        if(gateTest < gammaGate)
            gainMat(t,jj) = - log(lambdaFa) + normpdfLog(z,zBar,sMat) + log(pD); % Detections of already existing targets
            %gainMat(jj,t) =  normpdfLog(z,zBar,sMat); % Danchick/Newnam
        end
    end
    gainMat(t,m+t) = log(1-pD);  % Misdetections
end
for jj=1:m
    gainMat(n+jj,jj) = - log(lambdaFa) + log(muBirth) + log(pD);
    gainMat(n+jj,m+n+jj) = 0; 
end

   
% REWRITE THE WHOLE GODDAMN THING
% 
% 
% gainMat = -Inf*ones(m,n+2*m);  % Assignment is mapping from measurements to tracks
% for t=1:n
%     pBarArr(:,:,t) = fMat*pHatArr(:,:,t)*fMat' + qMat;
%     sArr(:,:,t) = hMat*pBarArr(:,:,t)*hMat' + rMat;
%     for jj=1:m
%         zBar = hMat*xBarArr(:,t);
%         z = measurements(:,jj);
%         innov = z-zBar;
%         sMat = sArr(:,:,t);
%         gateTest = innov'*(sMat\innov);
%         if(gateTest < gammaGate)
%             gainMat(jj,t) = - log(lambdaFa) + normpdfLog(z,zBar,sMat) + log(pD);% - log(1-pD);
%             gainMat(jj,t) =  normpdfLog(z,zBar,sMat); % Danchick/Newnam
%         end
%     end
% end
% for jj=1:m
%     gainMat(jj,n+jj) = 0;
%     gainMat(jj,n+jj) = log(lambdaFa)- log(pD) + log(1-pD);  % Danchick/Newnam
%     gainMat(jj,n+m+jj) = log(muBirth) -log(lambdaFa) + log(pD);
%     gainMat(jj,n+m+jj) = log(muBirth) - log(pD) + log(1-pD); % Danchick/Newnam
% end