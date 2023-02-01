function [probLogNewArr,xHypoNew,pHypoNew,xHParTracks,pHParTracks,zHParTracks] ...
    = processCalUReid(calU,calUCard,probLogHypos,xHPar,pHPar,z,lambda,pD,parents,pHatZero,mu,hMat,fMat,qMat,rMat,xHParTracks,pHParTracks,zHParTracks,k)


%h1 = [NaN,1,1,1,2]';
%h2 = [h1';NaN,NaN,NaN,NaN,1]';

%[iH1,inds1,score1] = hypothesisFind(h1,calU,calUCard,probLogHypos)
%[iH2,inds2,score2] = hypothesisFind(h2,calU,calUCard,probLogHypos)

k = size(calU,1)-1;

begs = tCloud2BegInd(calUCard);
ends = tCloud2EndInd(calUCard);
xHypoNew = NaN*zeros(4,sum(calUCard));
pHypoNew = NaN*zeros(10,sum(calUCard));
probLogNewArr = zeros(1,length(calUCard));

for iH=1:length(calUCard)
    
    omega = calU(:,begs(iH):ends(iH));
    
    if(~isempty(omega))
        xHCol = NaN*zeros(4,size(omega,2));
        pHCol = NaN*zeros(10,size(omega,2));
        zHCol = NaN*zeros(2,size(omega,2));
        
        uPi = sum(omega(end,:) == 0 & isnan(omega(end-1,:)));
        delta = sum(omega(end,:) > 0);
        d = sum(omega(end,:) > 0 & isnan(omega(end-1,:)));
        m = size(z,2);
        n = size(omega,2);
        nPrev = sum(~isnan(omega(end-1,:)));
        phi = m - delta;
        nLogTerms = zeros(1,n);
        
        xParH = xHPar(:,begs(iH):ends(iH));
        pParH = pHPar(:,begs(iH):ends(iH));
        
        
        for t=1:size(omega,2)
            
            % Check which of the four cases I00, I01, I10 or I11 we have
            
            if(omega(end,t) == 0)   % No measurement at current time step
                if(isnan(omega(end-1,t)))   % Track not established
                    
                    %xH = xHatZero;
                    %pH = pHatZero;
                    error('this kind of hypothesis should have been weeded out already');
                    
                else                        % Track already established
                    
                    xHatPrev = xParH(:,t);
                    pHatPrev = covVec2Mat(pParH(:,t));
                    xH = fMat*xHatPrev;
                    pH = fMat*pHatPrev*fMat' + qMat;
                    
                end
            else                    % Got measurement at current time step
                if(isnan(omega(end-1,t)))   % Track not established (Newborn observed target)
                    
                    zOmT = z(:,omega(end,t));
                    xH = [zOmT;zeros(2,1)];
                    pH = pHatZero;
                    zHCol(:,t) = zOmT;
                    
                else                        % Track already established
                    
                    zOmT = z(:,omega(end,t));
                    xHatPrev = xParH(:,t);
                    pHatPrev = covVec2Mat(pParH(:,t));
                    xBar = fMat*xHatPrev;
                    pBar = fMat*pHatPrev*fMat' + qMat;
                    sBar = hMat*pBar*hMat' + rMat;
                    kalmanGain = pBar*hMat'/sBar;
                    nu = zOmT - hMat*xBar;
                    xH = xBar + kalmanGain*nu;
                    pH = (eye(4) - kalmanGain*hMat)*pBar;
                    nLogTerms(t) = normpdfLog(zOmT,hMat*xBar,sBar);
                    zHCol(:,t) = zOmT;
                end
            end
            xHCol(:,t) = xH;
            pHCol(:,t) = covMat2Vec(pH);
        end
        xHypoNew(:,begs(iH):ends(iH)) = xHCol;
        pHypoNew(:,begs(iH):ends(iH)) = pHCol;
        
        xHParTracks(:,begs(iH):ends(iH),k+1) = xHCol;
        pHParTracks(:,begs(iH):ends(iH),k+1) = pHCol;
        zHParTracks(:,begs(iH):ends(iH),k+1) = zHCol;
        
        % Constituent terms in hypothesis probability
        
        %uTermLog = log(1/factorial(uPi));
        lambdaVTermLog = phi*log(lambda);
        kineTermLog = sum(nLogTerms);
        pDTermLog = log(pD)*(delta-d) + log(1-pD)*(n-delta-uPi);
        %muTermLog = log(mu)*(n-nPrev-d);
        muTermLog = 0;
        muVTermLog = d*log(mu);
        probLogParentTerm = probLogHypos(parents(iH));
        
        probLogNew = lambdaVTermLog + kineTermLog + pDTermLog + muTermLog + muVTermLog + probLogParentTerm;
        probLogNewArr(iH) = probLogNew;
        
%         if((iH==3 || iH==5) && ~isnan(iH1))
%            iH
%            probLogParentTerm
%            muVTermLog
%             probLogNew
%         end

%     if(k==4)
%        error('check here'); 
%     end
        
        
    else
        
        delta = sum(omega(end,:) > 0);
        m = size(z,2);
        n = size(omega,2);
        phi = m - delta;
        if(delta > 0 || n > 0)
           error('Got targets when I should not have targets'); 
        end
        lambdaVTermLog = phi*log(lambda);
        probLogParentTerm = probLogHypos(parents(iH));
        probLogNewArr(iH) = probLogParentTerm + lambdaVTermLog;
        
    end
end

