% Define the Bernoullis in the multi-Bernoulli

mu1 = 1;
mu2 = 3;
cv1 = 1;
cv2 = 0.45;
exi1 = 0.06;
exi2 = 0.8;

rateList = 0.01:0.02:0.26;
cvAppList = 0.1:0.2:4; % Covariance of approximant pdf
muApp = mu1;

nY1 = 104;
nY2 = nY1;

y1List = linspace(-2,6,nY1);
y2List = y1List;
dY = y1List(2)-y1List(1);


exiList = [exi1,exi2];
muList = [mu1,mu2];
cvList = cat(3,cv1,cv2);

%val = setdensityMultiBernoulli(2,exiList,muList,cvList)

pMonPart = zeros(2,nY1*nY2);
mapInds = zeros(2,nY1*nY2);
counter = 1;
for ii=1:nY1
    for jj=1:nY2
        pMonPart(:,counter) = [y1List(ii);y2List(jj)];
        mapInds(:,counter) = [ii;jj];
        counter = counter + 1;
    end 
end

kldArr = zeros(length(rateList),length(cvAppList));

for ii=1:length(rateList)
    for jj=1:length(cvAppList)
        rate = rateList(ii);
        cvApp = cvAppList(jj);
        
        vy1Log = log(rate) + normpdfLog(pMonPart(1,:),muApp,cvApp);
        vy2Log = log(rate) + normpdfLog(pMonPart(2,:),muApp,cvApp);
        
        pyLog = normpdfLog(y1List,mu1,cv1);
        gyLog = normpdfLog(y2List,mu2,cv2);
        vyLog = normpdfLog(y1List,muApp,cvApp) + log(rate);
        
        py1Log = normpdfLog(pMonPart(1,:),mu1,cv1);
        gy1Log = normpdfLog(pMonPart(1,:),mu2,cv2);
        py2Log = normpdfLog(pMonPart(2,:),mu1,cv1);
        gy2Log = normpdfLog(pMonPart(2,:),mu2,cv2);      
        
        denom = setdensityBernoulliPoisson([],exiList(2),mu2,cvList(:,:,2),rate,mu1,cvApp);
        
        inte0 = setdensityMultiBernoulli([],exiList,muList,cvList)*log(setdensityMultiBernoulli([],exiList,muList,cvList)/denom);
        
        inte1 = zeros(1,size(y1List,2));
        for qq=1:size(y1List,2)
            denom = setdensityBernoulliPoisson(y1List(:,qq),exiList(2),mu2,cvList(:,:,2),rate,mu1,cvApp);
            inte1(qq) = setdensityMultiBernoulli(y1List(:,qq),exiList,muList,cvList)...
                *log(setdensityMultiBernoulli(y1List(:,qq),exiList,muList,cvList)/denom);
        end
        integral1 = sum(inte1)*dY;
        
        inte2 = zeros(1,size(pMonPart,2));
        for qq=1:size(pMonPart,2)
           y = pMonPart(:,qq)';
           denom = setdensityBernoulliPoisson(y,exiList(2),mu2,cvList(:,:,2),rate,mu1,cvApp);
            inte2(qq) = setdensityMultiBernoulli(y,exiList,muList,cvList)...
                *log(setdensityMultiBernoulli(y,exiList,muList,cvList)/denom);            
        end
        integral2 = sum(inte2)*dY^2;
        

        kldArr(ii,jj) = inte0 + integral1 + integral2/2;
        if(~isreal(kldArr(ii,jj)))
            error('oi');
        end
        
        
    end
end

imagesc(rateList,cvAppList,kldArr');
axis xy;
colorbar;

% figure;
% plot(rateList,kldArr(:,10));
% 
% figure;
% plot(cvAppList,kldArr(6,:),'r');

% This seems to give the right minimum. How about Angel's expression for MB KLD bound?
% Could that expression possibly be exact?

kldAngArr = zeros(length(rateList),length(cvAppList));

for ii=1:length(rateList)
    for jj=1:length(cvAppList)
        rate = rateList(ii);
        cvApp = cvAppList(jj);
        
        %zerothPart = (1-exi1)*log(1-exi1) + exi1*log(exi1);
        zerothPart = (1-exi1)*log(1-exi1) + (1-exi1)*rate + exi1*log(exi1) + exi1*rate - exi1*log(rate);
        
        
        pyLog = normpdfLog(y1List,mu1,cv1);
        vyLog = normpdfLog(y1List,muApp,cvApp);
        
        inte1 = exi1*exp(pyLog).*(pyLog - vyLog);
        
        kldAngArr(ii,jj) = zerothPart + sum(inte1)*dY;
        
        
    end 
end

figure;
imagesc(rateList,cvAppList,kldAngArr');
axis xy;
colorbar;



