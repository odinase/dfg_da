clear all;
randState = 18;
randState = 21;
rand('state',randState);
randn('state',randState);
%set(0,'defaulttextinterpreter','none')
set(0,'defaulttextinterpreter','latex')
%set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'DefaultTextInterpreter', 'latex')
set(groot, 'defaultLegendInterpreter','latex');
rand(1,19000);
randn(1,19000);

% load('states160203.mat');
% 
% % Can I use measurements in world frame and still have a dynamic
% % surveillance region that moves with the ownship?
% 
% nV = size(stateArrHist,2);
% nVOthers = nV-1;
% nKSuperfine = size(stateArrHist,3);
% dTSuperfine = tSuperfineList(2)-tSuperfineList(1);
% 
% ownship = 1;
% targetships = setdiff(1:nV,ownship);

% Stuff from original pmbm program


scaleFactor = 3;
nK = 30;
%nK = 300;
%nK = 3;
faRate = 3.5*scaleFactor^2;
faRate = 1.5*scaleFactor^2;

sigmaV = 0.1;
sigmaR = 1;
dimTar = 4;
dimCov = dimTar+dimTar*(dimTar-1)/2;
dimZ = 2;
T = 1;

xMin = -40;
xMax = 20*scaleFactor;
yMin = -30;
yMax = 60*scaleFactor;


birthRateFactor = bump(1:nK,0,1,15,20,-1);



volPos = (xMax-xMin)*(yMax-yMin);

%
lambdaFa = faRate/volPos;




% Stuff from mechi generation program

pDList = [0.5,0.6,0.7,0.8,0.9];

% tPerK = 100;
% T = tPerK*dTSuperfine;

% stepSuperfineMin = 3400;
% stepSuperfineMax = 1.43e5;
% 
% chosenSteps = stepSuperfineMin:tPerK:(stepSuperfineMax+tPerK);
% chosenSteps(end) = [];

% nK = length(chosenSteps);
% axesMat = diag([1,0.5]);  % Target extent covariances

sigmaTheta = 2*pi/180;
sigmaR = 2;
% rMatPolar = diag([sigmaR^2,sigmaTheta^2]);
% 
% rMax = 100;

% %faRate = 40;
% areaCircle = rMax^2*pi;
% areaCircum = 4*rMax^2;
% faRateCircum = faRate*areaCircum/areaCircle;
% lambdaFa = faRateCircum/areaCircum;

%fig1 = figure;

rMaxSoFar = 0;


pInitVel = 2^2*eye(2);
pS = 0.99999;

nMC = 200;



scenario = struct('zList',{},'zIdentities',{},'zCard',{},'zLabels',{},'hTrueOrig',{},'hTrue',{},'targetsTrue',{});

%paramsMechi = struct('pD',pD,'T',T,'rMax',rMax,'areaCircle',areaCircle,'faRate',faRate,'lambdaFa',lambdaFa,'stateFullOwn',squeeze(stateArrHist(:,ownship,chosenSteps)));


params.PDList = pDList;
params.T = T;
params.rMax = NaN;
params.areaCircle = (xMax-xMin)*(yMax-yMin);
params.faRate = faRate;
params.lambdaFa = lambdaFa;
params.stateFullOwn = NaN;

params.axisDisp = NaN; 
params.pInitVel = pInitVel;
params.birthRateHistory = 1*ones(1,nK);
params.birthRateHistory = birthRateFactor;
params.pS = pS;
params.axisDisp = [xMin,xMax,yMin,yMax];


%sigmaV = 0.5;
qCont = sigmaV^2*eye(2);                            % Process noise strength for high-noise CV model

fCont = [zeros(2),eye(2);zeros(2,4)];
gCont = [zeros(2);eye(2)];

[fMat,qMat] = discretise_without_u(fCont,gCont,qCont,T);
qChol = chol(qMat)';
rMat = sigmaR^2*eye(2);
rChol = chol(rMat)';
fMat = eye(4) + [0,0,T,0;0,0,0,T;zeros(2,4)];
hMat = [eye(2),zeros(2,2)];


system = struct('fMat',{},'qMat',{},'hMat',{},'rCart',{},'rPol',{});
system(1).fMat = fMat;
system(1).qMat = qMat;
system(1).hMat = hMat;
system(1).rCart =rMat;
%system(1).birthRateFun = birthRateFun;
system(1).rPol = NaN;

targetsTrue = struct('id', {}, 'x', {}, 'k', {}, 'tLabel',{});


%error('ready to embark on for loops?');


for dd=1:length(pDList)
    pD = pDList(dd);
    for iMC=1:nMC
        iMC
        
        
        
        targetsTrue = struct('id', {}, 'x', {}, 'k', {}, 'tLabel',{});
        trueCard = zeros(1,nK);
        idTrueGen = 0;
        targetsTrueAlive = false(1,0); 
        
        % Simulation of true targets

        for k=1:nK

            % Propagate existing targets
            
            for t=1:length(targetsTrue)
                if(targetsTrueAlive(t))
                    
                    if(rand(1) < pS)
                        xPrev = targetsTrue(t).x(:,end);
                        xNew = fMat*xPrev + qChol*randn(dimTar,1);
                        targetsTrue(t).x = [targetsTrue(t).x,xNew];
                        targetsTrue(t).k = [targetsTrue(t).k,k];
                    else
                        targetsTrueAlive(t) = false; 
                    end
                end
            end

            % Newborn targets
            
            betaBorn = poissrnd(params.birthRateHistory(k));
            
            for t=1:betaBorn
                idTrueGen = idTrueGen + 1;
                newStatePos = [xMin + (xMax-xMin)*rand(1); yMin + (yMax-yMin)*rand(1)];
                newStateVel = chol(pInitVel)'*randn(2,1);
                x = [newStatePos;newStateVel];
                targetsTrue(end+1) = struct('id',idTrueGen,'x',x,'k',k,'tLabel',0);
                targetsTrueAlive = [targetsTrueAlive,true];
            end
            
            
            for t=1:length(targetsTrue)
                if(ismember(k,targetsTrue(t).k))
                    trueCard(k) = trueCard(k) + 1;
                end
            end
            
        end
        
        
                % Then I need to generate some measurements
        
        zList = zeros(dimZ,0);
        zIdentities = zeros(1,0);
        zCard = zeros(1,0);
        
        % Stuff to ensure that I can start with some tracks. TO BE REMOVED LATER
        
        %zK = [1,1;10,10]';
        %zList = [zList,zK];
        %zCard = [zCard,size(zK,2)];
        %zIdK = zeros(1,2);
        %zIdentities = [zIdentities,zIdK];
        
        maxM = max(3,max(zCard));
        
        for k=1:nK
            zK = zeros(dimZ,0);
            zIdK = zeros(1,0);
            for t=1:length(targetsTrue)
                kList = targetsTrue(t).k;
                kI = find(kList==k);
                if(~isempty(kI) && rand(1) < pD)
                    x = targetsTrue(t).x(:,kI);
                    z = hMat*x + rChol*randn(dimZ,1);
                    zK = [zK,z];
                    zIdK = [zIdK,targetsTrue(t).id];
                end
            end
            phiClutter = poissrnd(faRate);
            for ii=1:phiClutter
                zClutter = [xMin + (xMax-xMin)*rand(1); yMin + (yMax-yMin)*rand(1)];
                zK = [zK,zClutter];
                zIdK = [zIdK,0];
            end
            zList = [zList,zK];
            zIdentities = [zIdentities,zIdK];
            zCard = [zCard,size(zK,2)];
        end
        zBegs = tCloud2BegInd(zCard);
        zEnds = tCloud2EndInd(zCard);
        zLabels = 1:size(zList,2);
        
        
        
                % Construct the true hypothesis: New version
        
        hTrueOrig = NaN*zeros(nK,idTrueGen);
        for k=1:nK
            zIdK = zIdentities(zBegs(k):zEnds(k));
            for t=1:idTrueGen
                temp = find(zIdK == t);
                if(~isempty(temp))
                    hTrueOrig(k,zIdK(temp)) = temp;
                elseif(k >= targetsTrue(t).k(1) && k <= targetsTrue(t).k(end) )
                    hTrueOrig(k,t) = 0;
                else
                    hTrueOrig(k,t) = NaN;
                end
            end
        end
        nanColumns = all(isnan(hTrueOrig),1);
        hTrue = hTrueOrig(:,~nanColumns);
        nanCounts = sum(isnan(hTrue),1);
        [temp,ix] = sort(nanCounts,'ascend');
        hTrue = hTrue(:,ix);
        targetsTrue = targetsTrue(ix);
        
        
        for k=1:nK
           trueCardMC(k,iMC) = sum(~isnan(hTrue(k,:)));
        end
        
        
        scenario(iMC,dd) = struct('zList',zList,'zIdentities',zIdentities,'zCard',zCard,'zLabels',zLabels,'hTrueOrig',hTrueOrig,'hTrue',hTrue,'targetsTrue',targetsTrue);
        
        
        
        
        

        

        %zLabels = 1:length(zList);
        %scenario(iMC,dd) = struct('zList',zList,'zIdentities',zIdentities,'zCard',zCard,'zLabels',zLabels,'hTrueOrig',hTrue,'hTrue',hTrue,'targetsTrue',targetsTrue);
    end
end



