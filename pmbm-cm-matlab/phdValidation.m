function [lambdauInnerProds,gainMatLambdau] = phdValidation(phdTracks,measurements,predZLambdau,predSLambdau,gammaGate,inCol,pD,muPPP)



% Validation of PHD tracks - just do state estimation here as well?

m = size(measurements,2);
nTPHD = size(phdTracks,2);
gainMatLambdau = -Inf*ones(nTPHD+1,m);
for t=1:nTPHD
    for jj=1:m
        z = measurements(:,jj);
        zBar = predZLambdau(:,t);
        innov = z-zBar;
        sMat = predSLambdau(:,:,t);
        gateTest = innov'*(sMat\innov);
        if(gateTest < gammaGate)
            gainMatLambdau(t,jj) = log(pD) + log(phdTracks(inCol.exi,t)) + normpdfLog(z,zBar,sMat);
        end
    end
end
gainMatLambdau(nTPHD+1,:) = ones(1,m)*(log(pD) + log(muPPP));
lambdauInnerProds = zeros(1,m); % The inner products for new target weights for each current measurement
for jj=1:m
    lambdauInnerProds(jj) = sum(exp(gainMatLambdau(~isinf(gainMatLambdau(:,jj)),jj)));
end