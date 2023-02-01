%Demo with the implementation of the Poisson multi-Bernoulli mixture (PMBM)
%filter described in
%�. F. Garc�a-Fern�ndez, J. L. Williams, K. Granstr�m and L. Svensson, "Poisson Multi-Bernoulli Mixture Filter: Direct Derivation and Implementation," in IEEE Transactions on Aerospace and Electronic Systems, vol. 54, no. 4, pp. 1883-1901, Aug. 2018.

%Performance is measured with the GOSPA metric (alpha=2) and its
%decomposition into localisation errors for properly detected targets,
%and costs for missed targets and false targets (only possible for alpha=2).

%A. S. Rahmathullah, �. F. Garc�a-Fern�ndez and L. Svensson, "Generalized optimal sub-pattern assignment metric," 2017 20th International Conference on Information Fusion (Fusion), Xi'an, 2017, pp. 1-8.
% Short video on GOSPA
% https://www.youtube.com/watch?v=M79GTTytvCM


%Copyright (c) 2018, Angel F. Garcia-Fernandez
%All rights reserved.


% EFB revision 01: Make ready to include measuretments from Devijver example


clear
addpath('../GOSPA code')
addpath('../Assignment')

rand('seed',9)
randn('seed',9)
ScenarioWilliams15;


%load('../results210427a.mat');
load('../scenariosmall.mat'); % Contains scenario, params and system


% Initialise model
T = 1;
F = system.fMat;
Q = system.qMat;
H = system.hMat;
R =system.rCart;
chol_R=chol(R)';

p_s=params.pS;

xMin = params.axisDisp(1);
xMax = params.axisDisp(2);
yMin = params.axisDisp(3);
yMax = params.axisDisp(4);

Area=[xMax-xMin yMax-yMin];
nK = size(scenario(1,1).zCard,2);
Nsteps=nK; %Considered number of time steps in the simulation
l_clutter=params.faRate;
intensity_clutter=l_clutter/(Area(1)*Area(2));

%Birth
Ncom_b=1;
weights_b=0.005;
means_b=[(xMax-xMin)/2+xMin;(yMax-yMin)/2+yMin;0;0];
P_ini=diag([(xMax-xMin)^2/12 (yMax-yMin)^2/12 params.pInitVel(1) params.pInitVel(1) ]);

% I need to revise the Poisson component

nXPoiss = 6;
nYPoiss = 8;
xPoissList = linspace(xMin,xMax,nXPoiss);
yPoissList = linspace(yMin,yMax,nYPoiss);
dXPoiss = xPoissList(2)-xPoissList(1);
dYPoiss = yPoissList(2)-yPoissList(1);
nPoiss = nXPoiss*nYPoiss;





covs_b(1:4,1:4,1)=P_ini;
%Intensity Poisson prior (time 0)





Nx=4; %Single target state dimension
%Filter
T_pruning=0; %Threshold for pruning multi-Bernoulli mixtures weights
T_pruningPois=10^(-5); %Threshold for pruning PHD of the Poisson component
Nhyp_max=200;  %Maximum number of hypotheses (MBM components)
gating_threshold=20; %Threshold for gating
existence_threshold=0.00001; %Existence threshold: Bernoulli components with existence below this threshold are removed

type_estimator=3; %Choose Estimator 1, 2 or 3 as defined in the paper
existence_estimation_threshold1=0.4; %Only for esimator 1

%GOSPA errors for the estimator with highest hypothesis
squared_gospa_t_tot=zeros(1,Nsteps);
squared_gospa_loc_t_tot=zeros(1,Nsteps); %Localisation error
squared_gospa_false_t_tot=zeros(1,Nsteps); %False target error
squared_gospa_mis_t_tot=zeros(1,Nsteps); %Misdetection error



rand('seed',9)
randn('seed',9)

kInvestigate = 30;
nMC = size(scenario,1);

pBestAveDmc = zeros(length(kInvestigate),nMC,length(params.PDList));
pTrueaveDmc = zeros(length(kInvestigate),nMC,length(params.PDList));

ospaDmc = zeros(length(kInvestigate),nMC,length(params.PDList));
ospaLocDmc = zeros(length(kInvestigate),nMC,length(params.PDList));
ospaFalseDmc = zeros(length(kInvestigate),nMC,length(params.PDList));
ospaMisDmc = zeros(length(kInvestigate),nMC,length(params.PDList));

cardConsistencyRatioDmc = zeros(length(kInvestigate),nMC,length(params.PDList));

successOrFailureDmc = zeros(length(kInvestigate),nMC,length(params.PDList));
pTrueDmc = zeros(length(kInvestigate),nMC,length(params.PDList));

timesDmc = zeros(10,nK,nMC,length(params.PDList));

%We go through all Monte Carlo runs

%Nmc

for dd=1:length(params.PDList)
    dd
    p_d=params.PDList(dd);
    
    for i=1:Nmc
        tic
        
        error('hh');
        
        
        filter_pred.weightPois=lambda0;
        filter_pred.meanPois=means_b;
        filter_pred.covPois=covs_b;
        
        filter_pred.tracks=cell(0,1);
        filter_pred.globHyp=[];
        filter_pred.globHypWeight=[];
        N_hypotheses_t=zeros(1,Nmc);
        
        % Extract true targets information
        
        hTrue = scenario(i,dd).hTrue;
        numtruth = size(hTrue,2);
        X_truth = zeros(4*numtruth,nK);
        
        t_birth = zeros(1,numtruth);
        t_death = zeros(1,numtruth);
        
        
        
        for tt=1:numtruth
            kList = scenario(i,dd).targetsTrue(tt).k;
            for kI=1:length(kList)
                k = kList(kI);
                X_truth(4*tt-3:4*tt,k)=scenario(i,dd).targetsTrue(tt).x(:,kI);
            end
            t_birth(tt) = kList(1);
            t_death(tt) = kList(end)+1;
        end
        
        %error('opq');
        
        
        %Simulate measurements
        zList = scenario(i,dd).zList;
        zCard = scenario(i,dd).zCard;
        zBegs = tCloud2BegInd(zCard);
        zEnds = tCloud2EndInd(zCard);
        for k=1:Nsteps
            %z=CreateMeasurement(X_truth(:,k),t_birth,t_death,p_d,l_clutter,Area,k,H,chol_R,Nx);
            
            z = zList(:,zBegs(k):zEnds(k));
            
            
            
            
            z_t{k}=z;
        end
        
        %Perform filtering
        
        for k=1:Nsteps
            
            
            
            lambda0=params.birthRateHistory(k)/(prod(Area)); % EFB: I believe this is an intensity, and not a rate.
            
            
            %Update
            z=z_t{k};
            
            
            
            
            
            filter_upd=PoissonMBMtarget_update(filter_pred,z,H,R,p_d,k,gating_threshold,intensity_clutter,Nhyp_max);
            
            
            %filter_upd.globHyp
            
            
            
            
            
            
            
            %State estimation
            switch type_estimator
                case 1
                    X_estimate=PoissonMBMtarget_estimate1(filter_upd,existence_estimation_threshold1);
                case 2
                    X_estimate=PoissonMBMtarget_estimate2(filter_upd);
                case 3
                    X_estimate=PoissonMBMtarget_estimate3(filter_upd);
            end
            
            %Computation of squared GOSPA position error and its decomposition
            %Obtain ground truth state
            [squared_gospa,gospa_loc,gospa_mis,gospa_fal]=ComputeGOSPAerror(X_estimate,X_truth,t_birth,t_death,c_gospa,k);
            
            
            
            %We sum the squared errors
            squared_gospa_t_tot(k)=squared_gospa_t_tot(k)+squared_gospa;
            squared_gospa_loc_t_tot(k)=squared_gospa_loc_t_tot(k)+gospa_loc;
            squared_gospa_false_t_tot(k)=squared_gospa_false_t_tot(k)+gospa_fal;
            squared_gospa_mis_t_tot(k)=squared_gospa_mis_t_tot(k)+gospa_mis;
            
            %         %Draw filter output
            %         DrawFilterEstimates(X_truth,t_birth,t_death,X_estimate,[100,200],[100,200],z)
            %         pause(0.5)
            
            
            
            %Hypothesis reduction, pruning,normalisation
            filter_upd_pruned=PoissonMBMtarget_pruning(filter_upd, T_pruning,T_pruningPois,Nhyp_max,existence_threshold);
            filter_upd=filter_upd_pruned;
            
            N_hypotheses_t(k)=length(filter_upd.globHypWeight);
            
            %Prediction
            filter_pred=PoissonMBMtarget_pred(filter_upd,F,Q,p_s,weights_b,means_b,covs_b);
            
            
            if(ismember(k,kInvestigate))
                kIndex = find(kInvestigate == k);
                
                
                if(kIndex == 1)
                    
                    cardTrue = sum(~isnan(hTrue(k,:)));
                    
                    
%                     bestProb = max(cardsProb);
%                     trueProb = cardsProb(cardTrue+1);
%                     cardConsistencyRatioDmc(kIndex,iMC,dd) = trueProb/bestProb;
%                     
%                     pBestAveDmc(kIndex,iMC,dd) = bestProb;
%                     pTrueAveDmc(kIndex,iMC,dd) = trueProb;
                    
                    ospaDmc(kIndex,i,dd) = squared_gospa;
                    ospaLocDmc(kIndex,i,dd) = gospa_loc;
                    ospaFalseDmc(kIndex,i,dd) = gospa_fal;
                    ospaMisDmc(kIndex,i,dd) = gospa_mis;
                    
                    
                    
                    
                    
                    %error('stoiph');
                end
                
                
            end
            
            
        end
        
        t=toc;
        display(['Completed iteration number ', num2str(i),' time ', num2str(t), ' sec'])
        
    end
    
    
    
%     %Root mean square GOSPA errors at each time step
%     rms_gospa_t=sqrt(squared_gospa_t_tot/Nmc);
%     rms_gospa_loc_t=sqrt(squared_gospa_loc_t_tot/Nmc);
%     rms_gospa_false_t=sqrt(squared_gospa_false_t_tot/Nmc);
%     rms_gospa_mis_t=sqrt(squared_gospa_mis_t_tot/Nmc);
%     
%     
%     %Root mean square GOSPA errors across all time steps
%     rms_gospa_tot=sqrt(sum(squared_gospa_t_tot)/(Nmc*Nsteps))
%     rms_gospa_loc_tot=sqrt(sum(squared_gospa_loc_t_tot)/(Nmc*Nsteps))
%     rms_gospa_false_tot=sqrt(sum(squared_gospa_false_t_tot)/(Nmc*Nsteps))
%     rms_gospa_mis_tot=sqrt(sum(squared_gospa_mis_t_tot)/(Nmc*Nsteps))
end


% figure(1)
% plot(1:Nsteps,rms_gospa_t,'blue','Linewidth',1.3)
% grid on
% xlabel('Time step')
% ylabel('RMS GOSPA error')
% 
% figure(2)
% plot(1:Nsteps,rms_gospa_loc_t,'blue','Linewidth',1.3)
% grid on
% xlabel('Time step')
% ylabel('RMS GOSPA localisation error')
% 
% figure(3)
% plot(1:Nsteps,rms_gospa_false_t,'blue','Linewidth',1.3)
% grid on
% xlabel('Time step')
% ylabel('RMS GOSPA false target error')
% 
% figure(4)
% plot(1:Nsteps,rms_gospa_mis_t,'blue','Linewidth',1.3)
% grid on
% xlabel('Time step')
% ylabel('RMS GOSPA missed target error')
