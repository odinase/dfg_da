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

load('states160203.mat');

% Can I use measurements in world frame and still have a dynamic
% surveillance region that moves with the ownship?

nV = size(stateArrHist,2);
nVOthers = nV-1;
nKSuperfine = size(stateArrHist,3);
dTSuperfine = tSuperfineList(2)-tSuperfineList(1);

ownship = 1;
targetships = setdiff(1:nV,ownship);


pD = 0.6;

tPerK = 100;
T = tPerK*dTSuperfine;

stepSuperfineMin = 3400;
stepSuperfineMax = 1.43e5;

chosenSteps = stepSuperfineMin:tPerK:(stepSuperfineMax+tPerK);
chosenSteps(end) = [];

nK = length(chosenSteps);
axesMat = diag([1,0.5]);  % Target extent covariances

sigmaTheta = 2*pi/180;
sigmaR = 2;
rMatPolar = diag([sigmaR^2,sigmaTheta^2]);

rMax = 100;

faRate = 40;
areaCircle = rMax^2*pi;
areaCircum = 4*rMax^2;
faRateCircum = faRate*areaCircum/areaCircle;
lambdaFa = faRateCircum/areaCircum;

fig1 = figure;

rMaxSoFar = 0;




nMC = 100;

scenario = struct('zList',{},'zIdentities',{},'zCard',{},'zLabels',{},'hTrueOrig',{},'hTrue',{},'targetsTrue',{});

paramsMechi = struct('pD',pD,'T',T,'rMax',rMax,'areaCircle',areaCircle,'faRate',faRate,'lambdaFa',lambdaFa,'stateFullOwn',squeeze(stateArrHist(:,ownship,chosenSteps)));

targetsTrue = struct('id', {}, 'x', {}, 'k', {}, 'tLabel',{});

for iMC=1:nMC
    iMC
    
    zList = zeros(2,0);
    zCard = zeros(1,0);
    hTrue = zeros(nK,nVOthers);
    
    for tt=1:length(targetships)
        
        x = squeeze(stateArrHist(1:2,targetships(tt),chosenSteps));
        
        % But I should also have world-frame velocity in true target state!
        
        
        
        velocities = zeros(2,size(x,2));
        velocities(:,1) = rotmat2d(eulerHist(3,ownship,chosenSteps(1)))*vels(1:2,chosenSteps(1),targetships(tt));
        velocities(:,2:end) = diff(x,[],2);
      
        x = [x;velocities];
        
        k = 1:length(chosenSteps);
        tLabel = tt;
        id = tt;
        targetsTrue(tt) = struct('id', id, 'x', x, 'k', k, 'tLabel',tLabel);
        
        
    end
    
    zIdentities = zeros(1,0);
    for k=1:nK
        posOwn2D = stateArrHist(1:2,ownship,chosenSteps(k));
        measurements = zeros(2,0);
        meaLabels = zeros(1,0);
        for tt=1:length(targetships)
            ii = targetships(tt);
            rn = rand(1);
            if(rn < pD)
                rotmatI = rotmat2d(eulerHist(3,ownship,chosenSteps(k)));
                rMatExt = rotmatI*axesMat*rotmatI';
                posTar2D = stateArrHist(1:2,ii,chosenSteps(k));
                deltaPos = posTar2D - posOwn2D + chol(rMatExt)'*randn(2,1);
                deltaPosPolar = c2p(deltaPos);
                zPolar = deltaPosPolar + chol(rMatPolar)'*randn(2,1);
                zCart = posOwn2D + p2c(zPolar);
                measurements = [measurements,zCart];
                meaLabels = [meaLabels,tt];
                

                
                
                
                if(zPolar(1) > rMaxSoFar)
                    rMaxSoFar = zPolar(1);
                end
                
            end
            if(k==14 && tt==2)
                %error('check that measurement location makes sense');
            end
        end
        
        % How about clutter measurements?
        % Would be most convenient for what I want to demonstrate here to just keep them spatially uniform in the Cartesian coordinates
        
        nClutterCircum = poissrnd(faRateCircum);
        clutterCircum = randn(2,nClutterCircum)*2*rMax-rMax;
        inCircle = clutterCircum(1,:).^2+clutterCircum(2,:).^2 <= rMax^2;
        clutter = clutterCircum(:,inCircle) + repmat(posOwn2D,[1,sum(inCircle)]);
        measurements = [measurements,clutter];
        meaLabels = [meaLabels,zeros(1,size(clutter,2))];
        [temp,ix] = sort(measurements(1,:),'ascend');
        
        measurements = measurements(:,ix);
        meaLabels = meaLabels(:,ix);
        
        for tt=1:nVOthers
            q = find(meaLabels == tt);
            if(~isempty(q))
                hTrue(k,tt) = q;
            end
        end
        zIdentities = [zIdentities,meaLabels];
        zList = [zList,measurements];
        zCard = [zCard,size(measurements,2)];
        
        
        if(nMC ==1)
            plot(measurements(2,:),measurements(1,:),'*b');
            axis([posOwn2D(2)-150,posOwn2D(2)+150,posOwn2D(1)-150,posOwn2D(1)+150]);
            title(num2str(k));
            pause(0.1);
        end
        
        
    end
    zLabels = 1:length(zList);
    scenario(iMC) = struct('zList',zList,'zIdentities',zIdentities,'zCard',zCard,'zLabels',zLabels,'hTrueOrig',hTrue,'hTrue',hTrue,'targetsTrue',targetsTrue);
end




