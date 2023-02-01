% Devijver PMBM versions
% Version 01 based on script_reid_devijver_ho17.m
% Version 04: Try to do validation/gain calculation, prior hypothesis management and track filtering in the appropriate order.
% Version 05: Now that should be OK. Proceed to do the posterior hypothesis expansion using Murty.
% Version 06: Reduce Gain Matrix to PMBM style
% Version 09: Ensure that no tracks are shared between clusters whatsoever (also including newly established tracks)
% Finally manage to run through all time steps, although with singular covariances.
% Version 12: Now I should finally do some pruning.
% Version 13: Looks very promising. Seems to work
% Version 14: Need to start visualizing.
% Version 19: I also need to keep track of the truth assignments
% Version 20: First version made on new MBP
% Version 21: Makes sense at least until k=4.
% Version 24: When do cardinalities in a single cluster begin to vary?
% Version 28: Try to include basic cluster splitting
% Version 30: Try cluster splitting on TAES example
% Version 32: Attempt to deal with empty hypotheses
% Version 33: Works for all time steps and results look sensible. Am I ready to look at success rates now?
% Version 34: Try to include success rates analysis.
% PMBM with PHD versions
% Version 06: Try to combine track filtering and likelihood calcuation in same phase
% Version 09: Able to maintain true hypothesis to the end and assign it
% decent probability. Uploaded to mycloud.
% Version 11: Seems to work well, including cardinality estimation
% Version 12: Fixed some bugs and now cardinality success array makes much more sense.
% Version 13: Now I also need the evaluated probability of the true cardinality. Or was it the best cardinality?
% Version 14: Can I managed to satisfy Devijver inequalities for cardinality?
% Version 15: Devijer finally appears to be working for the very sparse scenario
% Version 16: Successfully validated Devijver for various tuning parameters. Overly slow when nK = 10.
% Version 20: Include option for semi-two-point initialization
% Version 24: To avoid explosion in cardinality evaluation try to put it all in the same for loop which then can be interrupted
% Version 26: Get death probability into ground truth model as well
% Version 27: Also include an xy-scale factor
% Version 28: Movie from version 28 now looks properly decoupled
% Version 34: Pre-allocation and separation into functions for Double Murty and Cluster Splitting has significantly reduced runtime. Still there are some spikes in run-time plots.
% Version 36: Attempt to use Brita Gades idea to further split away tracks that claim bulk of probability mass
% Version 38: Do the conncomp in a while loop to ensure sufficiently small superclusters
% Version 41: Implement consensual merging. I really hope this will curb the complexity.
% Version 42: Instead of consensual merging, attempt to make the wile-loop more elegant and efficient.
% Version 44: I think I have all the booking in place to implement the heuristic merging scheme. Now try to implement it.
% Version 45: Try bona-fide branch-and-bound hypothesis generation
% Version 46: Use Murty for the first phase of combining top hypotheses
% Version 48: Branch and bound while loop seems to work
% Version 50: Branch and bound now in separate function
% Version 51: Bugs fixed in B-and-B function. Now it seems to work for the entire time step 8. Commented out DM approach.
% Version 52: First version where B-and-B seems to work flawlessly.
% Version 54: Type-3 estimator implemented
% Version 55: GOSPA included
% Version 58: Include a dynamic birth rate to make cardinality success rates more meaningful
% Version 59: I should loop over PD from 0.3 to 0.9 to check for potential cardinality inconsistency as function of PD around 0.55
% Version 60c: I need to clean up time-taking
% Version 61: Separate data generation so that I can do this both for my and Angels versions
% Version 61: Systematic loading of Mechi og Devijver data
% Version 64: Seems to give excellent results on Mechi for high PD
% Version 66: Set aside for studying what happens with GGX at k 465.
% Version 68: Attempt to correct the inter-cluster assignment so that it satisfies at-most-one assumptions
% Version 69: Use V3 Branch-and-bound at k=8. GGX is missing cluster support.
% Version 70: Try to use V3 Branch-and-bound from beginning
% Version 71: Try to include recycling of all significant gainMatFull elements that fail to make it into MBM.
% Version 72: For now it seems reasonable to divide the given lambdaFa by 10 or so. Set aside in case I want to work more on the assignment problem at k=2. (mechiLowFAk2-AfterRec-Unsorted.xlsx sheet 2)
% Version 73: See if this problem is solved by means of post-DM recycling. Not entire successful. But without it, perfect at least until k=9 (could have explored more hypotheses though)
% Version 75: Include benchmarking with Angels method
% Version 76: Go back to the track initialization scenario for comparing success rates between DMCM and Angel
% Version 77 Include optional storage of results to enable MC simulations over several days.
% Version 79: Fix edge-break stuff that appearently has been wrong all the way
% Version 81: Be more fair in comparison with Angel wrt hypothesis count etc
% Version 82: Improve time-taking so that I get clearer picture of run-time
% Version 86: Include warm start after kLast by reading from file
% Version 87: Start replacing a2bcFaster with a2bcFasterFaster. Still Mechi.
% Version 88: Look at track initialization scenario for M-best analysis

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

addpath('PMBM filter');
addpath('GOSPA code');
addpath('Assignment');

cList = struct('cMat',{});


doAngel = false;
doRunnall = false;
doSemiTwoPoint = true;
doTAES = false;
doPlot =false;
doMovie = false;
doMechi = true;
%moviename = 'mechiTrackingMovie08MechiFocusFirst60TimeSteps';
moviename = '9ravens9';
%moviename = 'initscenario01';
%moviename = 'mechiTrackingMovie10SteroAdjusted02AllTargetsAllTimeSteps';
doFilter = false; % If false, the script does only data generation
doShowPHD = true;
doSuccessRates = true; % Whether I should do success rate evaluation
doPostDMRecycle = false;
doWarmstart = true;
warmstartString = '../pmbm_large_files/warmstart600.mat';

kInvestigateList = zeros(1,0);
%ise = evalin( 'base', 'exist(''savedID'',''var'') == 1' );

ise = isfile('../pmbm_large_files/tempFileCM12.mat');
if(~ise)
    
    if(doMechi)
        load('../pmbm_large_files/scenarioMechi2PD.mat');
        
        
        params.lambdaFa = params.lambdaFa/10;
        params.faRate = params.faRate/10;
        params.pInitVel = 100*eye(2);
        system.qMat = system.qMat*1.6^2;
        %scenario = scenario(1,2);
        params.PDList = params.PDList(2);
        scenario = scenario(:,2);
        %params.PDList = params.PDList(2);        
        
    else
        %load('scenarioInit100b.mat');
        %load('scenarioInit10a.mat');
        %load('scenarioInit1000.mat');
        load('../pmbm_large_files/scenarioInit200a.mat');
        
    end
    
    

    nMC = 1;
    %nMC = 100;
    %nMC = 1;
    
    plotMC = 1;
    adoleThres = 1;
    nMC = size(scenario,1);
    finalMC = 20;
    nHypoMax = 30;
    nHypoTotalMax = 150; % Maximal number of hypotheses allowed per cluster.
    %nHypoTotalMax = 1000;
    %nHypoTotalMax = 6;
    
    mRunnMax = 8;
    sig = 0.05; % Significance level for majority tracks in cluster splitting
    %sig = 0.00000005;
    max_iter_auction = 1000;
    nK = length(params.birthRateHistory);
    nD = size(scenario,2);
    
    dimTar = size(scenario(1,1).targetsTrue(1).x,1);
    dimZ = size(scenario(1,1).zList,1);
    
    gGate = 5;
    gGate = 3;
    gammaGate = gGate^2;
    
    
    % HO-MHT stuff
    
    bMax = 3;
    hypoPruneThreshold = 8;
    
    % TO-MHT STUFF
    
    trackPruneThreshold = 6;
    nSigma = 5;
    gammaVar = nSigma^2;
    maxSigma = 50;
    splittingThreshold = 0.018; % Tracks with track probability below this threshold may be removed in cluster splitting
    mahaCSThres = 4; % Setting this one to zero disables Mahalanobis-thresholding in the CS phase
    nonSplitLag = 5;
    nonSplitLag = 2; % Must be lower for Mechi data because of the strong dynamics
    
    % generateZS = @(xStuff,pStuff) genZSFun(xStuff,pStuff,hMat,rMat);
    % generateXP = @(xBar,sBar,pBar,nuM) genXPFun(xBar,sBar,pBar,nuM,hMat);
    
    % Transponder stuff
    
    nTexi = 2; % Number of transponder targets known to exist
    xTexiPriorList1 = [-30,-20,1,1]';
    xTexiPriorList2 = [10,50,-1,-1]';
    xTexiPriorList = [xTexiPriorList1,xTexiPriorList2];
    pTexiInit = eye(4);
    
    % Enumeration analysis
    
%     exploreAnalysis = struct('probMassHInD',{},'probMassDInH',{},'probMassHInP',{},'probMassPInH',{},'nHypoTotalMax',{},...
%         'nCap',{},'mCap',{},'hypoCountH',{},'hypoCountD',{},'hypoCountP',{},'nCL',{},'priorAveSize',{},'pqLen',{},...
%         'scoreDMedianDiff',{},'scoreDMedianRatio',{},'scorePMedianDiff',{},'scorePMedianRatio',{});
    
    exploreAnalysis = struct('nHypoTotalMax',{},'nCap',{},'mCap',{},'hypoCountM',{},'hypoCountH',{},...
        'hypoCountD',{},'hypoCountDS',{},...
        'hypoCountP',{},'hypoCountPS',{},'nCL',{},'priorAveSize',{},'pqLen',{},...
        'pMassMNotH',{},'pMassMNotD',{},'pMassMNotDS',{},'pMassMNotP',{},'pMassMNotPS',{},...
        'pRatioD',{},'pRatioDS',{},'pRatioP',{},...
        'dCountRatioH',{},'dSCountRatioH',{},'pCountRatioH',{});
    
    
    preExplore = struct('hypos',{},'hyposCard',{},'clusters',{},'clustersCard',{},'probLogHypos',{},'iC',{},'assocLocal',{},'gainMatPostC',{},...
        'indicesOfNewbornTracks',{},'nHypoTotalMax',{},'trackNumberLookup',{},'k',{});
    
    
    %branchAndBoundExplore6(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoTotalMax,trackNumberLookup,k);
    
%     exploreAnalysis(end+1) =  struct('nHypoTotalMax',nHypoTotalMax,'nCap',nCap,'mCap',mCap,...
%         'hypoCountM',size(probLogLocalM,2),'hypoCountH',size(probLogLocal,2),'hypoCountD',size(probLogLocalD,2),'hypoCountDS',size(probLogLocalDS,2),...
%         'hypoCountP',size(probLogLocalP,2),'hypoCountPS',size(probLogLocalPS,2),'nCL',nCLD,'priorAveSize',mean(pcSizeD),'pqLen',pqLen,...
%         'pMassMNotH',pMassMNotH,'pMassMNotD',pMassMNotD,'pMassMNotDS',pMassMNotDS,...
%         'pMassMNotP',pMassMNotP,'pMassMNotPS',pMassMNotPS);
%     
    
    
    % Visualization stuff
    
    trackProbLimitDisp = 0.001;
    
    if(doMechi)
        load('colorsMechi.mat');
        
    else
        colorArr = repmat([1,0,0;... % OS
            0.1,0.86,0.95;... % NA
            0.35,0.35,0.6;... % ST
            0,0.65,0.65;... % NK
            0.78,0,0.95;... % WK
            0.85,0.2,0.2;...   % DR
            0.6,0.2,0.2;...   % TZ
            0.75,0.8,0;...   % KK 8
            0.8,0.5,0.5;... % SS 9
            0.9,0.6,0.2;... % NP 10
            0,0,0;... % GW 11
            1,0.6,0;... % AM 12
            0.2,0.1,0.4;... % MA 13
            1,0,0.8;... % AF 14
            0,0,1;... % GK 15
            1,0.86,0;... % KH 16
            0,0.9,0;... % TJ 17
            0.25,0.35,1],[78,1]); % NW 18
    end
    
    
    colorsClusters = repmat([1,0.85,0.85;... % OS
        1,0.86,0;... % KH 16
        1,0,0.8;... % AF 14
        0,0.9,0;... % TJ 17
        0.8,0.8,1;... % GK 15
        0.8,0.5,0.5;... % SS 9
        0.9,0.6,0.2;... % NP 10
        0.75,0.8,0;...   % KK 8
        0.1,0.86,0.95;... % NA
        0.35,0.35,0.6;... % ST
        0,0.65,0.65;... % NK
        0.78,0,0.95;... % WK
        0.85,0.2,0.2;...   % DR
        0.6,0.2,0.2;...   % TZ
        0,0,0;... % GW 11
        1,0.6,0;... % AM 12
        0.2,0.1,0.4;... % MA 13
        0.25,0.35,1],[78,1]); % NW 18
    
    
    
    
    % Stuff to make statistics over MC simulations
    
    
    pDList = params.PDList;
    
    maxCard = 105;
    cards = 0:maxCard;
    
    cardsDmc = zeros(maxCard+1,nK,nMC,nD);
    successOrFailureArr = zeros(nK,nMC,nD);
    
    
    hyposCountMC = zeros(nK,nMC,nD);
    maxHyposCountMC = zeros(nK,nMC,nD);
    hyposProbMC = zeros(1,nMC,nD);
    trueCardMC = zeros(nK,nMC,nD);
    phdAveCardMC = zeros(nK,nMC,nD);
    mapCardMC = zeros(nK,nMC,nD);
    cardsProbMC = zeros(maxCard+1,nMC,nK,nD);
    hyposCountPerCluster = zeros(50,nMC,nK,nD);
    hyposCountStagesMC = zeros(4,nK,nMC,nD);
    
    % Should also keep track of maximal hypothesis size in each cluster
    
    
    maxHypoSizePerCluster = zeros(50,nMC,nK,nD);
    
    
    cardsC2CMC = zeros(maxCard+1,nMC,nK); % Histograms over how many clusters go to a new cluster in Double Murty
    
    
    % Stuff to measure run-time
    
    timesArr = zeros(10,nMC,nD);
    timesInSplitting = zeros(10,nK,nMC);
    innerSplittingComplexity = zeros(50,nMC,nK);
    timesArrExt = zeros(10,nK,nMC,nD);
    
    nTracksTentaMC = zeros(nK,nMC,nD);
    timesOtherExt = zeros(1,nMC);
    
    % GOSPA stuff
    
    squared_gospa_MC =zeros(nK,nMC);
    squared_gospa_loc_MC =zeros(nK,nMC); %Localisation error
    squared_gospa_false_MC =zeros(nK,nMC); %False target error
    squared_gospa_mis_MC =zeros(nK,nMC); %Misdetection error
    
    % GT and data stuff for storage
    
    
    
    kInvestigate = 1:nK; % Only do stuff like cardinality evaluation at selected k's to speed up run-time
    
    % Arrays to store the PD-related investigations ("dmc" stuff)
    
    
    
    
    pBestAveDmc = zeros(length(kInvestigate),nMC,length(pDList));
    pTrueAveDmc = zeros(length(kInvestigate),nMC,length(pDList));
    
    ospaDmc = zeros(length(kInvestigate),nMC,length(pDList));
    ospaLocDmc = zeros(length(kInvestigate),nMC,length(pDList));
    ospaFalseDmc = zeros(length(kInvestigate),nMC,length(pDList));
    ospaMisDmc = zeros(length(kInvestigate),nMC,length(pDList));
    
    cardConsistencyRatioDmc = zeros(length(kInvestigate),nMC,length(pDList));
    
    successOrFailureDmc = zeros(length(kInvestigate),nMC,length(pDList));
    successOrFailureDmcAngel = zeros(length(kInvestigate),nMC,length(pDList));
    pTrueDmc = zeros(length(kInvestigate),nMC,length(pDList));
    
    timesDmc = zeros(10,nK,nMC,length(pDList));
    
    timesPreDmc = zeros(2,nMC,length(pDList));
    
    
    trueTrackTTPHistCM = NaN*zeros(nK,size(scenario(1,1).hTrue,2),length(pDList),nMC);
    if(doAngel)
        
        trueTrackTTPHistA = NaN*zeros(nK,size(scenario(1,1).hTrue,2),length(pDList),nMC);
        
        pBestAveAngel = zeros(length(kInvestigate),nMC,length(pDList));
        pTrueAveAngel = zeros(length(kInvestigate),nMC,length(pDList));
        cardConsistencyRatioAngel = zeros(length(kInvestigate),nMC,length(pDList));
        
        
        ospaDmcAngel = zeros(nK,nMC,length(pDList));
        ospaLocDmcAngel = zeros(nK,nMC,length(pDList));
        ospaFalseDmcAngel = zeros(nK,nMC,length(pDList));
        ospaMisDmcAngel = zeros(nK,nMC,length(pDList));
        
        
    end
    
    %nMC = 1;
    
    iMCBeg = 1;
    ddBeg = 1;
    ddEnd = 5;
    
else
    
    load('../pmbm_large_files/tempFileCM11.mat');
    iMCBeg = savedID(1);
    ddBeg = savedID(2);
    
    disp(['Loading stuff at iMC ',num2str(iMC),' dd ',num2str(dd)]);
    
    
end
%finalMC = 1;


if(doWarmstart)
    doPlotThis =doPlot;
    doMovieThis = doMovie;
    doWarmstartThis = true;
    nHypoTotalMaxA = nHypoTotalMax;
    
   %load('../pmbm_large_files/mechi610h400.mat'); 
   load(warmstartString); 
    ddBeg = dd;
    iMCBeg = iMC;
    kBeg = kLast + 1;
    doPlot = doPlotThis;
    doMovie = doMovieThis;
    doWarmstart = doWarmstartThis;
    
    nHypoTotalMax = nHypoTotalMaxA ;
    
    'entered warmstart'
    
else
    
   kBeg = 1; 
end

'before dd-loop'
nHypoTotalMax


for dd=ddBeg:ddEnd
    dd
    
    
    pD = pDList(dd);
    pS = params.pS;
    
    c_gospa=10; %Parameter c of the GOSPA metric. We also consider p=2 and alpha=2
    %c_gospa = 3;
    %c_gospa = 30;
    
    
    if(doAngel)
        N_hypotheses_t=zeros(1,nMC);
        Nhyp_max=800;  %Maximum number of hypotheses (MBM components)
        Nhyp_max=400;
        gating_threshold=20; %Threshold for gating
        T_pruning=0; %Threshold for pruning multi-Bernoulli mixtures weights
        T_pruningPois=10^(-5); %Threshold for pruning PHD of the Poisson component
        existence_threshold=0.00001; %Existence threshold: Bernoulli components with existence below this threshold are removed
        
        type_estimator=3; %Choose Estimator 1, 2 or 3 as defined in the paper
        existence_estimation_threshold1=0.4; %Only for esimator 1
        
        %GOSPA errors for the estimator with highest hypothesis
        squared_gospa_t_Angel=zeros(nK,nMC);
        squared_gospa_loc_t_Angel=zeros(nK,nMC); %Localisation error
        squared_gospa_false_t_Angel=zeros(nK,nMC); %False target error
        squared_gospa_mis_t_Angel=zeros(nK,nMC); %Misdetection error
        
        

        X_estimate_hist = zeros(40,nK);
        
        %cardsArr(:,:,iMC)
        cardsArrAngel = zeros(maxCard,nK,nMC);
        
    end
    for iMC=iMCBeg:finalMC
        
        disp(['PD value ',num2str(pD),' Monte Carlo run ',num2str(iMC)]);
        tic
        
        %iMC
        if(mod(iMC,1) == 0)
            iMC
        end
        
        % ---------------------------------------------------------------------
        % Preparations for simulation
        % ---------------------------------------------------------------------
        
        
        t1 = clock;
        
        if(doPlot && iMC == plotMC)
            fig1 = figure;
            if(doMechi)
                set(fig1,'position',[200,200,1220,510]);
            else
                set(fig1,'position',[200,200,420,610]);
            end
            if(doMovie)
                v = VideoWriter(moviename,'MPEG-4');
                v.FrameRate = 4;
                open(v);
            end
        end
        
        %randState = iMC + 3000001;
        randState = 170 + iMC;
        randState = 1708 + iMC;
        %randState = 1708 + 8;
        %randState = 1708+27;
        %randState = 1708+4;
        %randState = 1708+507;
        %randState = 1708+88; % Used up to version 51
        %randState = 1708+79;f
        %randState = 205;
        %randState = 1866;
        %randState = 1731;
        
        
        rand('state',randState);
        randn('state',randState);
        %set(0,'defaulttextinterpreter','none')
        rand(1,19000);
        randn(1,19000);
        
        % Simulate
        
        
        
        targetsTrue = struct('id', {}, 'x', {}, 'k', {}, 'tLabel',{});
        trueCard = zeros(1,nK);
        idTrueGen = 0;
        targetsTrueAlive = false(1,0);
        
        
        % Row indices in track file
        
        tarXInCol = 1:dimTar;
        tarPInCol = dimTar+1:dimTar+dimTar+dimTar*(dimTar-1)/2;
        meaLastInCol = dimTar+dimTar+dimTar*(dimTar-1)/2+1; % Only index of measurement in current scan
        costInCol = meaLastInCol(end)+1;
        contribInCol = costInCol+1;
        exiInCol = contribInCol+1;
        visiInCol = exiInCol+1;
        labelInCol = visiInCol+1; % BUT IS THIS TRACK LABEL OR TARGET LABEL?
        lastInCol = labelInCol+1;
        inCol = struct('tarX',tarXInCol,'tarP',tarPInCol,'meaLast',meaLastInCol,'cost',costInCol,'contrib',contribInCol,...
            'exi',exiInCol,'visi',visiInCol,'label',labelInCol,'last',lastInCol);
        
        % And also historical state estimates with covariances
        
        dimTarXP = tarPInCol(end);
        estHistCol = zeros(dimTarXP,0,0);
        
        
        hTrue = scenario(iMC,dd).hTrue;
        
        if(doTAES)
            
            z1 = [1,1]';
            z2 = [2,2; 2,4]';
            z3 = [3,3]';
            z4 = [4,2; 4,4]';
            labels1 = 1;
            labels2 = [2,3];
            labels3 = [4];
            labels4 = [5,6];
            zList = [z1,z2,z3,z4];
            labelArr = [labels1,labels2,labels3,labels4];
            zCard = [size(z1,2),size(z2,2),size(z3,2),size(z4,2)];
            zBegs = tCloud2BegInd(zCard);
            zEnds = tCloud2EndInd(zCard);
            hTrue = [1,1,1,2; NaN,2,0,1]';
            
        elseif(~doWarmstart)
            
            for k=1:nK
                trueCardMC(k,iMC,dd) = sum(~isnan(hTrue(k,:)));
            end
            
        end
        
        %----------------------------------------------------------------------
        % New stuff
        %----------------------------------------------------------------------
        
        % Hypotheses are now simply collections of track file elements
        
        trackFile = zeros(inCol.last,0);
        maxLag = 8;
        meaHistCol = zeros(maxLag,0); % This is in some sense the real track file
        trackFileShadow = zeros(inCol.last,0,maxLag); % 3D extension of track file to store historical estimates etc.
        
        meaHistFull = zeros(maxLag,0);
        
        
        clusters = zeros(1,0);
        clustersCard = zeros(1,0);  % We start with ONE empty cluster
        hypos = zeros(1,0);
        hyposCard = zeros(1,0);
        probLogHypos = zeros(1,0);   % Logarithmic probability of original hypothesis.
        muPPP = 0;
        trueNextParent = 1;
        labelGen = 0;
        
        mu1PHD = [-30,-40,0,0]';
        mu2PHD = [60,20,0,0]';
        cov1PHD = blkdiag(30^2*eye(2),6^2*eye(2));
        w1PHD = 0.1;
        w2PHD = 0.05;
        
        phdTrack1 = zeros(inCol.last,1);
        phdTrack1(inCol.tarX) = mu1PHD;
        phdTrack1(inCol.tarP) = covMat2Vec(cov1PHD);
        phdTrack1(inCol.meaLast) = NaN;
        phdTrack1(inCol.exi) = w1PHD;
        phdTrack1(inCol.cost) = NaN;
        phdTrack1(inCol.contrib) = NaN;
        phdTrack1(inCol.visi) = NaN;
        phdTrack1(inCol.label) = NaN;
        phdTrack1(inCol.last) = NaN;
        
        phdTrack2 = zeros(inCol.last,1);
        phdTrack2(inCol.tarX) = mu2PHD;
        phdTrack2(inCol.tarP) = covMat2Vec(cov1PHD);
        phdTrack2(inCol.meaLast) = NaN;
        phdTrack2(inCol.exi) = w1PHD;
        phdTrack2(inCol.cost) = NaN;
        phdTrack2(inCol.contrib) = NaN;
        phdTrack2(inCol.visi) = NaN;
        phdTrack2(inCol.label) = NaN;
        phdTrack2(inCol.last) = NaN;
        
        phdTracks = [phdTrack1,phdTrack2];
        phdTracks = zeros(size(phdTracks,1),0);
        %meaHistCol = [meaHistCol,NaN*zeros(maxLag,2)];
        
        t2 = clock;
        deltaT = etime(t2,t1);
        timesArr(1,iMC,dd) = deltaT;
        timesArrExt(1,1,iMC,dd) = deltaT;
        
        
        zBegs = tCloud2BegInd(scenario(iMC,dd).zCard);
        zEnds = tCloud2EndInd(scenario(iMC,dd).zCard);
        
        
        % Determine pruning threshold in pre-clustering
        % Main criterion: I should be able to have, say 4 misdetections after two-poing initialization
        
        allowedCountMisdet = 3;
        nKPT = allowedCountMisdet+2;
        xInitPT = zeros(dimTar,1);
        pInitPT = blkdiag(system.rCart,params.pInitVel);
        
        xPTDetArr = zeros(dimTar,nKPT);
        pPTDetArr = zeros(dimTar,dimTar,nKPT);
        xPTMisdetArr = zeros(dimTar,nKPT);
        pPTMisdetArr = zeros(dimTar,dimTar,nKPT);
        
        xPTDetArr(:,1) = xInitPT;
        xPTMisdetArr(:,1) = xInitPT;
        pPTDetArr(:,:,1) = pInitPT;
        pPTMisdetArr(:,:,1) = pInitPT;
        
        
        pPTPred = system.fMat*pPTDetArr(:,:,1)*system.fMat' + system.qMat;
        sMat = system.hMat*pPTPred*system.hMat' + system.rCart;
        kalmanGain = pPTPred*system.hMat'/sMat;
        pPTDetArr(:,:,2) = (eye(dimTar) - kalmanGain*system.hMat)*pPTPred;
        pPTMisdetArr(:,:,1) = pPTDetArr(:,:,2);
        
        exiDetArr = ones(1,nKPT);
        exiMisdetArr = zeros(1,nKPT);
        exiMisdetArr(1:2) = ones(1,2);
        
        
        scoreDetArr = zeros(1,nKPT);
        scoreMisdetArr = zeros(1,nKPT);
        
        volPos = params.areaCircle;
        lambdaFa = params.faRate/volPos;
        muBirth = params.birthRateHistory(1)/volPos;
        predExi = params.pS;
        
        scoreDetArr(2) = - log(lambdaFa + pD*muBirth) + normpdfLog(zeros(2,1),zeros(2,1),sMat) + log(pD) + log(predExi);
        scoreMisdetArr(2) = - log(lambdaFa + pD*muBirth) + normpdfLog(zeros(2,1),zeros(2,1),sMat) + log(pD) + log(predExi);
        
        for kPT=3:nKPT
            
            % With detection
            
            pPTPred = system.fMat*pPTDetArr(:,:,kPT-1)*system.fMat' + system.qMat;
            sMat = system.hMat*pPTPred*system.hMat' + system.rCart;
            kalmanGain = pPTPred*system.hMat'/sMat;
            pPTDetArr(:,:,kPT) = (eye(dimTar) - kalmanGain*system.hMat)*pPTPred;
            predExi = params.pS;
            scoreDetArr(kPT) = scoreDetArr(kPT-1)- log(lambdaFa + pD*muBirth) + normpdfLog(zeros(2,1),zeros(2,1),sMat) + log(pD) + log(predExi);
            
            % Without detection
            
            pPTPred = system.fMat*pPTMisdetArr(:,:,kPT-1)*system.fMat' + system.qMat;
            sMat = system.hMat*pPTPred*system.hMat' + system.rCart;
            pPTMisdetArr(:,:,kPT) = pPTPred;
            predExi = exiMisdetArr(kPT-1)*params.pS;
            exiMisdetArr(kPT) = predExi;
            scoreMisdetArr(kPT) = scoreMisdetArr(kPT-1) + log(1-predExi + predExi*(1-pD));
        end
        
        preClusterThreshold = abs(scoreDetArr(end)-scoreMisdetArr(end));
        if(preClusterThreshold < 1 || preClusterThreshold > 20)
            %warning('Beware of unreasonable value of preClusterThreshold');
        end
        preClusterThresholdOrig = preClusterThreshold;
        
        % ---------------------------------------------------------------------
        % Preparation for Garcia PMBM
        % ---------------------------------------------------------------------
        
        if(doAngel)
            
            if(doMechi)
                xMin = params.stateFullOwn(1,1) - params.rMax;
                xMax = params.stateFullOwn(1,1) + params.rMax;
                yMin = params.stateFullOwn(2,1) - params.rMax;
                yMax = params.stateFullOwn(2,1) + params.rMax;
            else
                xMin = params.axisDisp(1);
                xMax = params.axisDisp(2);
                yMin = params.axisDisp(3);
                yMax = params.axisDisp(4);
            end
            
            
            Ncom_b=1;
            
            means_b=[(xMax-xMin)/2+xMin;(yMax-yMin)/2+yMin;0;0];
            P_ini=diag(4^2*[(xMax-xMin)^2/12 4^2*(yMax-yMin)^2/12 params.pInitVel(1) params.pInitVel(1) ]);
            covs_b(1:4,1:4,1)=P_ini;
            
            volPos = params.areaCircle;
            lambda0=params.birthRateHistory(1)/(prod(volPos)); % EFB: I believe this is an intensity, and not a rate.
            
            
            % Additional filter_upd stuff to handle flat P-component
            
            
            gaussPeakVal2D = 1/(sqrt(det(P_ini(1:2,1:2)))*(2*pi)^(1));
            weights_b=lambda0/gaussPeakVal2D;
            
            
            % filter_upd stuff from Angel
            
            filter_upd.weightPois=weights_b;
            filter_upd.meanPois=means_b;
            filter_upd.covPois=covs_b;
            
            filter_upd.tracks=cell(0,1);
            filter_upd.globHyp=[];
            filter_upd.globHypWeight=[];
            
            
            % Extract true targets information
            
            hTrue = scenario(iMC,dd).hTrue;
            numtruth = size(hTrue,2);
            X_truth = zeros(4*numtruth,nK);
            
            t_birth = zeros(1,numtruth);
            t_death = zeros(1,numtruth);
            
            
            
            for tt=1:numtruth
                kList = scenario(iMC,dd).targetsTrue(tt).k;
                for kI=1:length(kList)
                    k = kList(kI);
                    X_truth(4*tt-3:4*tt,k)=scenario(iMC,dd).targetsTrue(tt).x(:,kI);
                end
                t_birth(tt) = kList(1);
                t_death(tt) = kList(end)+1;
            end
            
            
            
            
        end
        % ---------------------------------------------------------------------
        % End of simulation preparations
        % ---------------------------------------------------------------------
        
        
        
        timesPreDmc(1,iMC,dd) = toc;
        
        if(doWarmstart) 
            
            kBegThis = kBeg;
          dPlotThis = doPlot;
           doMovieThis = doMovie;            
           nHypoTotalMaxA = nHypoTotalMax;
           
           load(warmstartString);  
           doWarmstart = false;
           
           nHypoTotalMax = nHypoTotalMaxA;
           doPlot = doPlotThis;
           doMovie = doMovieThis;
           kBeg = kBegThis;
           
           %nHypoTotalMax = 300;
        end
        %error('jkjj');
        
        
        'before k-loop'
        nHypoTotalMax
        
        
        for k=kBeg:nK
%             if(k==85)
%                 
%                error('kkk'); 
%             end
            
            tic
            
            
            measurements = scenario(iMC,dd).zList(:,zBegs(k):zEnds(k));
            m = size(measurements,2);
            nTracks = size(trackFile,2);
            
            
            ta = clock;
            
            tb = clock;
            timesOtherExt(iMC) = timesOtherExt(iMC)+ etime(t2,t1);
            
            if(doMechi)
                volPos = params.areaCircle;
                
                scaleDisp = 0.6;
                nXB = 210;
                nYB = 140;
                xMinDisp = params.stateFullOwn(1,k) - scaleDisp*params.rMax*2;
                xMaxDisp = params.stateFullOwn(1,k) + scaleDisp*params.rMax*2;
                yMinDisp = params.stateFullOwn(2,k) - scaleDisp*params.rMax + 20;
                yMaxDisp = params.stateFullOwn(2,k) + scaleDisp*params.rMax + 20;
                
                
                %                 xyc = scenario(iMC,dd).targetsTrue(8).x(1:2,k);
                %
                %                 xMinDisp = xyc(1) - 60;
                %                 xMaxDisp = xyc(1) + 60;
                %                 yMinDisp = xyc(2) - 30;
                %                 yMaxDisp = xyc(2) + 30;
                
            else
                volPos = (params.axisDisp(2)-params.axisDisp(1))*(params.axisDisp(4)-params.axisDisp(3));
                
                % PHD grid parameters
                
                nXB = 210;
                nYB = 140;
                xMinDisp = params.axisDisp(1) - 10;
                xMaxDisp = params.axisDisp(2) + 10;
                yMinDisp = params.axisDisp(3) - 10;
                yMaxDisp = params.axisDisp(4) + 10;
            end
            lambdaFa = params.faRate/volPos;
            
            if(doShowPHD)
                xB= linspace(xMinDisp,xMaxDisp,nXB);
                yB = linspace(yMinDisp,yMaxDisp,nYB);
                phdGrid = zeros(nXB,nYB);
                pMonB = zeros(2,nXB*nYB);
                mapIndsB = zeros(2,nXB*nYB);
                counter = 1;
                for jj=1:nYB
                    for ii=1:nXB
                        pMonB(:,counter) = [xB(ii);yB(jj)];
                        mapIndsB(:,counter) = [ii;jj];
                        counter = counter + 1;
                    end
                end
            end
            muBirth = params.birthRateHistory(k)/volPos;
            
            if(nMC==1 || mod(k,10) == 0)
                k
            end
            %k
            
            
            if(doAngel)
                
                if(doMechi)
                    xMinA = params.stateFullOwn(1,k) - params.rMax;
                    xMaxA = params.stateFullOwn(1,k) + params.rMax;
                    yMinA = params.stateFullOwn(2,k) - params.rMax;
                    yMaxA = params.stateFullOwn(2,k) + params.rMax;
                else
                   xMinA = xMin;
                   xMaxA = xMax;
                   yMinA = yMin;
                   yMaxA = yMax;
                    
                    
                end
                means_b=[(xMaxA-xMinA)/2+xMinA;(yMaxA-yMinA)/2+yMinA;0;0];
                covs_b(1:4,1:4,1)=diag(4^2*[(xMaxA-xMinA)^2/12 4^2*(yMaxA-yMinA)^2/12 params.pInitVel(1) params.pInitVel(1) ]);
                
                
                
                %fuwp = filter_upd.weightPois
                
                lambda0=params.birthRateHistory(k)/(prod(volPos));
                filter_pred=PoissonMBMtarget_pred(filter_upd,system.fMat,system.qMat,pS,weights_b,means_b,covs_b);
                
                %fpwp = filter_pred.weightPois
                
                [filter_upd,cListM]=PoissonMBMtarget_update_rPC(filter_pred,measurements,pD,k,gating_threshold,lambdaFa,Nhyp_max,doMechi,system,params);
                
                cList(end+1:end+length(cListM)) = cListM;
                
%                 if(k> 10 && isempty(filter_upd.globHyp))
%                    error('check filter_upd'); 
%                 end
                
                
                % Follow Angel in calculating OSPA before pruning.
                
                switch type_estimator
                    case 1
                        X_estimate=PoissonMBMtarget_estimate1(filter_upd,existence_estimation_threshold1);
                    case 2
                        X_estimate=PoissonMBMtarget_estimate2(filter_upd);
                    case 3
                        X_estimate=PoissonMBMtarget_estimate3(filter_upd);
                end
                X_estimate_hist(1:length(X_estimate),k) = X_estimate;
                
                [squared_gospa,gospa_loc,gospa_mis,gospa_fal]=ComputeGOSPAerror(X_estimate,X_truth,t_birth,t_death,c_gospa,k);
                
                
                
                
                
                %We sum the squared errors
                squared_gospa_t_Angel(k,iMC)=squared_gospa_t_Angel(k,iMC)+squared_gospa;
                squared_gospa_loc_t_Angel(k,iMC)=squared_gospa_loc_t_Angel(k,iMC)+gospa_loc;
                squared_gospa_false_t_Angel(k,iMC)=squared_gospa_false_t_Angel(k,iMC)+gospa_fal;
                squared_gospa_mis_t_Angel(k,iMC)=squared_gospa_mis_t_Angel(k,iMC)+gospa_mis;
                
                
                
                
                ospaDmcAngel(k,iMC,dd) = squared_gospa_t_Angel(k);
                ospaLocDmcAngel(k,iMC,dd) = squared_gospa_loc_t_Angel(k);
                ospaFalseDmcAngel(k,iMC,dd) = squared_gospa_false_t_Angel(k);
                ospaMisDmcAngel(k,iMC,dd) = squared_gospa_mis_t_Angel(k);
                
%                 if(k==635)
%                     error('Check Angel just before pruning');
%                 end
                
                %Hypothesis reduction, pruning,normalisation
                filter_upd_pruned=PoissonMBMtarget_pruning(filter_upd, T_pruning,T_pruningPois,Nhyp_max,existence_threshold);
                filter_upd=filter_upd_pruned;
                N_hypotheses_t(k)=length(filter_upd.globHypWeight);
                [hyposA,hyposCardA,meaHistColA,trackWNumbers,trackHypoNumbers,ttpsA,probsA,trackFileA] = angelToEdmund(filter_upd,scenario(iMC,dd).zCard,k,inCol,maxLag);
                ttpsOfTrueTracks = trueTracksTTP(ttpsA,meaHistColA,hTrue,k);
                trueTrackTTPHistA(k,1:size(ttpsOfTrueTracks,2),dd,iMC) = ttpsOfTrueTracks;
                
                % Cardinality stuff
                
                [X_estimate2,pcard_tot,N_tracks]=PoissonMBMtarget_estimate2(filter_upd);
                cardsArrAngel(1:N_tracks+1,k,iMC) = pcard_tot';
                %cardsArrAngel(1,k,iMC) = pcard_tot(end);
                
                %error('stop here');
%                 if(k==215)
%                    error('kk'); 
%                 end
                
            end
            
            % -------------------------------------------------------------
            % Prediction phase --------------------------------------------
            % -------------------------------------------------------------
            
            t1 = clock;
            
            muPPP = muPPP*(1-pD)*pS + muBirth;
            
            % But I should also predict the GM portion on the PHD
            
            nTPHD = size(phdTracks,2);
            phdTracks(inCol.exi,:) = pS*(1-pD)*phdTracks(inCol.exi,:);
            predXLambdau = system.fMat*phdTracks(inCol.tarX,:);
            predZLambdau = system.hMat*predXLambdau;
            predPLambdau = zeros(dimTar,dimTar,nTPHD);
            predSLambdau = zeros(dimZ,dimZ,nTPHD);
            prevP = covVec2Mat(phdTracks(inCol.tarP,:));
            for ii=1:nTPHD
                predPLambdau(:,:,ii) = system.fMat*prevP(:,:,ii)*system.fMat' + system.qMat;
                if(doMechi)
                    own2tar = predXLambdau(1:2,ii) - params.stateFullOwn(1:2,k);
                    [~,Ra] = cmAlternative(c2p(own2tar),system.rPol);
                    rMat = system.rCart + Ra;
                else
                    rMat = system.rCart;
                end
                
                predSLambdau(:,:,ii) = system.hMat*predPLambdau(:,:,ii)*system.hMat' + rMat;
            end
            phdTracks(inCol.tarX,:) = predXLambdau;
            phdTracks(inCol.tarP,:) = covMat2Vec(predPLambdau);
            
            
            
            begsHypo = tCloud2BegInd(hyposCard);
            endsHypo = tCloud2EndInd(hyposCard);
            
            % Before I do anything more on hypotheses I should expand the
            % track file according to current measurement set
            
            predX = system.fMat*trackFile(inCol.tarX,:);
            predP = zeros(dimTar,dimTar,nTracks);
            prevP = covVec2Mat(trackFile(inCol.tarP,:));
            predS = zeros(dimZ,dimZ,nTracks);
            predZ = system.hMat*predX;
            for ii=1:nTracks
                
                if(det(prevP(:,:,ii)) < eps)
                    error('Singular s-matrix');
                end
                predP(:,:,ii) = system.fMat*prevP(:,:,ii)*system.fMat' + system.qMat;
                
                
                if(doMechi)
                    own2tar = predX(1:2,ii) - params.stateFullOwn(1:2,k);
                    [~,Ra] = cmAlternative(c2p(own2tar),system.rPol);
                    rMat = system.rCart + Ra;
                else
                    rMat = system.rCart;
                end
                predS(:,:,ii) = system.hMat*predP(:,:,ii)*system.hMat' + rMat;
            end
            
            % Also predict existence probabilities in the MBM component
            
            for t=1:nTracks
                exiPrev = trackFile(inCol.exi,t);
                exiNew = pS*exiPrev;
                trackFile(inCol.exi,t) = exiNew;
            end
            
            t2 = clock;
            deltaT = etime(t2,t1);
            timesArr(2,iMC,dd) = timesArr(2,iMC,dd) + deltaT;
            timesArrExt(2,k,iMC,dd) = deltaT;
            
            timesDmc(1,k,iMC,dd) = toc;
            
            
            tic
            
            % -------------------------------------------------------------
            % Validation phase --------------------------------------------
            % -------------------------------------------------------------
            
            %[trackSumProbsV,trackTotalProbsV] = trackProbAccumulate(meaHistCol,trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
            
            
            if(max(hypos) > size(trackFile,2))
                error('hypos refers to non-existent tracks');
            end
            
            t1 = clock;
            %
            
            [lambdauInnerProds,gainMatLambdau] = phdValidation(phdTracks,measurements,predZLambdau,predSLambdau,gammaGate,inCol,pD,muPPP);


%             % Validation of PHD tracks - just do state estimation here as well?
%             
%             nTPHD = size(phdTracks,2);
%             gainMatLambdau = -Inf*ones(nTPHD+1,m);
%             for t=1:nTPHD
%                 for jj=1:m
%                     z = measurements(:,jj);
%                     zBar = predZLambdau(:,t);
%                     innov = z-zBar;
%                     sMat = predSLambdau(:,:,t);
%                     gateTest = innov'*(sMat\innov);
%                     if(gateTest < gammaGate)
%                         gainMatLambdau(t,jj) = log(pD) + log(phdTracks(inCol.exi,t)) + normpdfLog(z,zBar,sMat);
%                     end
%                 end
%             end
%             gainMatLambdau(nTPHD+1,:) = ones(1,m)*(log(pD) + log(muPPP));
%             lambdauInnerProds = zeros(1,m); % The inner products for new target weights for each current measurement
%             for jj=1:m
%                 lambdauInnerProds(jj) = sum(exp(gainMatLambdau(~isinf(gainMatLambdau(:,jj)),jj)));
%             end
            
            % Validation of Bernoulli tracks
            
            
            [meaHistColOld,trackNumberLookup,predX,predZ,predP,predS,gainMatFull,hypos,hyposCard,trackFile,trackFileShadow,meaHistCol,newTrackSubs,nTracksTenta] = bernoulliValidation(trackFile,meaHistCol,measurements,lambdauInnerProds,pD,inCol,doSemiTwoPoint,...
                predX,predZ,predP,predS,prevP,gammaGate,lambdaFa,hypos,hyposCard,clusters,clustersCard,probLogHypos,trackFileShadow,k,adoleThres);
            
            
            t2 = clock;
            deltaT = etime(t2,t1);
            timesArr(3,iMC,dd) = timesArr(3,iMC,dd) + deltaT;
            timesArrExt(3,k,iMC,dd) = deltaT;
            
            
            timesDmc(2,k,iMC,dd) = toc;
            tic
            
            nTracksTentaMC(k,iMC,dd) = nTracksTenta;
            
            
            % -------------------------------------------------------------
            % Track filtering phase
            % -------------------------------------------------------------
            
            t1 = clock;            
            
            [trackFileNew,trackFileShadowNew,meaHistColNew,trackParents,indicesOfNewbornTracks] = trackFiltering(predX,predZ,predP,predS,newTrackSubs,measurements,system,inCol,pD,trackFile,gainMatFull,gainMatLambdau,...
                muPPP,doMechi,params,k,lambdauInnerProds,labelGen,maxLag,trackFileShadow,meaHistCol,predZLambdau,predSLambdau,predPLambdau);

            
            t2 = clock;
            deltaT = etime(t2,t1);
            timesArr(4,iMC,dd) = timesArr(4,iMC,dd) + deltaT;
            timesArrExt(4,k,iMC,dd) = deltaT;            
            
            % In future version include option to merge tracks which claim the same measurements over last maxLag scans.
            
%             proximities = zeros(nTracks,nTracks);
%             labelShare = zeros(nTracks,nTracks);
%             for ii=1:nTracks
%                 pI = covVec2Mat(trackFile(inCol.tarP,ii));
%                 labelI = trackFile(inCol.label,ii);
%                 for jj=(ii+1):nTracks
%                     pJ = covVec2Mat(trackFile(inCol.tarP,jj));
%                     innovIJ = trackFile(inCol.tarX,ii)-trackFile(inCol.tarX,jj);
%                     tTest = innovIJ'*((pI+pJ)\innovIJ);
%                     if(tTest < dimTar^2)
%                         proximities(ii,jj) = 1;
%                     end
%                     labelJ = trackFile(inCol.label,jj);
%                     if(labelI == labelJ)
%                         labelShare(ii,jj) = 1;
%                     end
%                 end
%             end
%             proximities = (proximities + proximities') > 0;
%             labelShare = (labelShare + labelShare') > 0;
            

            
            timesDmc(3,k,iMC,dd) = toc;
            tic
            
            % Now I have established the new track file.
            % Then proceed to new hypothesis collection.
            
%                 if(k> 10 && isempty(filter_upd.globHyp))
%                    error('check filter_upd'); 
%                 end            
            
            
            hyposCountStagesMC(1,k,iMC,dd) = length(hyposCard);
            
            % --------------------------------------------------------------
            % --------------------------------------------------------------
            % Combine clustering and hypothesis exploration in one
            % --------------------------------------------------------------
            % --------------------------------------------------------------
            
            t1 = clock;
            
            [assocLocal,gainMatPostC,masters] = clusteringPreprocess(hypos,hyposCard,clusters,clustersCard,gainMatFull,meaHistCol,m,preClusterThreshold);
            
            
            t2 = clock;
            deltaT = etime(t2,t1);
            timesArr(5,iMC,dd) = timesArr(5,iMC,dd) + deltaT;
            timesArrExt(5,k,iMC,dd) = deltaT;
            
            timesDmc(4,k,iMC,dd) = toc;
            
            tic
            
            % -------------------------------------------------------------------------
            % Perform Murty-based clustering based on reduced cluster-measurement graph
            % ... combined with Murty-based generation of new hypotheses
            % -------------------------------------------------------------------------
            
            t1 = clock;
            
%             [x,ia] = unique(clusterAssoc);
%             masters = clusterAssoc(ia);
%             
%             assocLocal = zeros(2,size(clusterAssoc,2));
%             assocLocal(1,:) = clusterAssoc;
%             assocLocal(2,masters) = 1;
            

            for ii=1:length(masters)
                c = length(find(assocLocal(1,:)==masters(ii)));
                cardsC2CMC(c,iMC,k) = cardsC2CMC(c,iMC,k) + 1;
            end
            
            % Containers for new hypotheses and clusters
            
            nClusters = length(masters);
            newHypos = zeros(1,0); % To contain track numbers for each of the new hypotheses after clustering
            newHyposCard = zeros(1,0);
            newProbLogs = zeros(1,0);
            cWNew = zeros(1,0);
            cCardWNew = zeros(1,0);
            newHyposCount = 0;
            
            tfClusterMembership = NaN*zeros(1,size(meaHistColNew,2));  % Which new cluster a track belongs to
            
            
            [shareTracks,shareClusters] = testClustersShareTracks(clusters,clustersCard,hypos,hyposCard);
            if(~isempty(shareTracks))
                error('tracks shared among clusters before pruning c');
            end
            
            %[probs,aSel,b,c] = probLogs2ProbabilitiesForHypos(clusters,probLogHypos,clusters,clustersCard,size(hyposCard,2));
            
            
            % [trackSumProbs,trackTotalProbs] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
            
            
            [hyposNewA,hyposCardNewA,clustersNewA,clustersCardNewA,probabilitiesA,probLogsNewA] = sortHyposInCluster(hypos,hyposCard,clusters,clustersCard,probLogHypos);

            [boolsH,boolsT] = testRepeatedMeasurements(hypos,hyposCard,meaHistCol);
            if(any(boolsH))
                error('Did I get repeated measurements in a single hypothesis before DM?');
            end
            
            if(k==614)
                %lims = [xMinDisp,xMaxDisp,yMinDisp,yMaxDisp];
                %temp = plotBeforeDoubleMurty(hypos,hyposCard,clusters,clustersCard,trackFile,trackFileShadow,inCol,meaHistCol,predX,predZ,predP,phdTracks,muPPP,measurements,scenario,system,colorArr,lims,nXB,nYB,k,iMC,dd);
            end
            
            for iC=1:size(masters,2)
                
                t1BB = clock;
                %[hyposLocal,hyposCardLocal,probLogLocal] = branchAndBoundExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoTotalMax,trackNumberLookup,k);
                
                % Alternative branch and bound explore to be written here in main script first.
                
                if(k < 1)
                    
                    [hyposLocal,hyposCardLocal,probLogLocal] = branchAndBoundExplore3(hypos,hyposCard,clusters,clustersCard,probLogHypos,...
                        iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoTotalMax,trackNumberLookup,k);
                    
                else
                    
                     %  'before'
                    [hyposLocal,hyposCardLocal,probLogLocal,kInvesti,pqLen] = branchAndBoundExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoTotalMax,trackNumberLookup,k);
                    kInvestigateList = [kInvestigateList,kInvesti];
                    
                    %'after'
                    
                    % Find the top combinations of prior hypotheses
                    % Then check quality of K-best output for 
                    % * Low value of K
                    % * High value of K
                    % * The ceiling approach
                    % I find the N best prior hypotheses, where M is the default value for N 
                    
%                     nCap = 30;
%                     mCap = 30;
%                     [hyposLocalD,hyposCardLocalD,probLogLocalD,kInvestiD,nCLD,pcSizeD] = doubleMurtyExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoTotalMax,trackNumberLookup,nCap,mCap,k);
%                     
%                     [hyposLocalP,hyposCardLocalP,probLogLocalP,kInvestiP,nCLP,pcSizeP] = proptoExplore(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoTotalMax,trackNumberLookup,nCap,mCap,k);
%                     
%                     nHypoVeryMany = 1000;
%                     [hyposLocalM,hyposCardLocalM,probLogLocalM,kInvestiM,pqLenM] = branchAndBoundExplore6(hypos,hyposCard,clusters,clustersCard,probLogHypos,iC,assocLocal,gainMatPostC,indicesOfNewbornTracks,nHypoMax,nHypoVeryMany,trackNumberLookup,k);
%                     
%                     % New exploration analysis
%                     
%                     % Make alternatives of D and P with only 150 hypotheses
% 
%                     [probLogSortedD,ix] = sort(probLogLocalD,'descend');
%                     nHDS = min(length(ix),length(hyposCardLocal));
%                     probLogLocalDS = probLogSortedD(1:nHDS);
%                     hyposCardLocalDS = hyposCardLocalD(ix(1:nHDS));
%                     [aRemove,tCloudRemove] = pickIndC(ix(1:nHDS),hyposCardLocalD);
%                     hyposLocalDS = hyposLocalD(aRemove);                                        
%                     
%                     [probLogSortedP,ix] = sort(probLogLocalP,'descend');
%                     nHPS = min(length(ix),length(hyposCardLocal));
%                     probLogLocalPS = probLogSortedP(1:nHPS);
%                     hyposCardLocalPS = hyposCardLocalP(ix(1:nHPS));
%                     [aRemove,tCloudRemove] = pickIndC(ix(1:nHPS),hyposCardLocalP);
%                     hyposLocalPS = hyposLocalP(aRemove);   
%                     
%                     % Analyize missing probability mass versus 1000 hypos
%                     
%                     foundInBranchboundMD = NaN*zeros(1,length(probLogLocalD));
%                     foundInBranchboundMP = NaN*zeros(1,length(probLogLocalP));                    
%                     foundInBranchboundMDS = NaN*zeros(1,length(probLogLocalDS));
%                     foundInBranchboundMPS = NaN*zeros(1,length(probLogLocalPS));                                        
%                     foundInBranchboundMH = NaN*zeros(1,length(probLogLocal));                                                            
%                     
%                     begsLocalH = tCloud2BegInd(hyposCardLocal);
%                     endsLocalH = tCloud2EndInd(hyposCardLocal);
%                     begsLocalM = tCloud2BegInd(hyposCardLocalM);
%                     endsLocalM = tCloud2EndInd(hyposCardLocalM);                    
%                     begsLocalD = tCloud2BegInd(hyposCardLocalD);
%                     endsLocalD = tCloud2EndInd(hyposCardLocalD);    
%                     begsLocalP = tCloud2BegInd(hyposCardLocalP);
%                     endsLocalP = tCloud2EndInd(hyposCardLocalP);                      
%                     begsLocalDS = tCloud2BegInd(hyposCardLocalDS);
%                     endsLocalDS = tCloud2EndInd(hyposCardLocalDS);    
%                     begsLocalPS = tCloud2BegInd(hyposCardLocalPS);
%                     endsLocalPS = tCloud2EndInd(hyposCardLocalPS);                                          
%                     
%                     foundInD = NaN*zeros(1,length(probLogLocalM));
%                     foundInP = NaN*zeros(1,length(probLogLocalM));                    
%                     foundInDS = NaN*zeros(1,length(probLogLocalM));
%                     foundInPS = NaN*zeros(1,length(probLogLocalM));                    
%                     foundInH = NaN*zeros(1,length(probLogLocalM));  
%                     
%                     for ii=1:length(hyposCardLocalM)
%                         hI = hyposLocalM(begsLocalM(ii):endsLocalM(ii));
%                         for jj=1:length(hyposCardLocalD)
%                             hD = hyposLocalD(begsLocalD(jj):endsLocalD(jj));
%                             if(all(ismember(hI,hD)) && isnan(foundInD(ii))) % All tracks in hI are also in hD
%                                 foundInD(ii) = jj;
%                             end                               
%                         end
%                         for jj=1:length(hyposCardLocalDS)
%                             hDS = hyposLocalDS(begsLocalDS(jj):endsLocalDS(jj));
%                             if(all(ismember(hI,hDS)) && isnan(foundInDS(ii))) % All tracks in hI are also in hD
%                                 foundInDS(ii) = jj;
%                             end                               
%                         end
%                         for jj=1:length(hyposCardLocalP)
%                             hP = hyposLocalP(begsLocalP(jj):endsLocalP(jj));
%                             if(all(ismember(hI,hP)) && isnan(foundInP(ii))) % All tracks in hI are also in hD
%                                 foundInP(ii) = jj;
%                             end                               
%                         end                        
%                         for jj=1:length(hyposCardLocalPS)
%                             hPS = hyposLocalPS(begsLocalPS(jj):endsLocalPS(jj));
%                             if(all(ismember(hI,hPS)) && isnan(foundInPS(ii))) % All tracks in hI are also in hD
%                                 foundInPS(ii) = jj;
%                             end                               
%                         end    
%                         for jj=1:length(hyposCardLocal)
%                             hH = hyposLocal(begsLocalH(jj):endsLocalH(jj));
%                             if(all(ismember(hI,hH)) && isnan(foundInH(ii))) % All tracks in hI are also in hD
%                                 foundInH(ii) = jj;
%                             end                               
%                         end                         
%                     end
%                     
%                     probabilitiesM = exp(probLogLocalM);
%                     probabilitiesM = probabilitiesM/sum(probabilitiesM);
%                     
%                     probsMNotH = probabilitiesM(isnan(foundInH));
%                     probsMNotD = probabilitiesM(isnan(foundInD));
%                     probsMNotDS = probabilitiesM(isnan(foundInDS));
%                     probsMNotP = probabilitiesM(isnan(foundInP));
%                     probsMNotPS = probabilitiesM(isnan(foundInPS));                    
%                     
%                     pMassMNotH = sum(probsMNotH);
%                     pMassMNotD = sum(probsMNotD);
%                     pMassMNotDS = sum(probsMNotDS);
%                     pMassMNotP = sum(probsMNotP);
%                     pMassMNotPS = sum(probsMNotPS);
%                     
%                     % Comparison of best score values
%                     
%                     foundInDRelH = NaN*zeros(1,length(probLogLocal));
%                     foundInPRelH = NaN*zeros(1,length(probLogLocal));                    
%                     foundInDSRelH = NaN*zeros(1,length(probLogLocal));
%                     foundInPSRelH = NaN*zeros(1,length(probLogLocal));                    
%                                         
%                     for ii=1:length(hyposCardLocal)
%                         hI = hyposLocal(begsLocalH(ii):endsLocalH(ii));
%                         for jj=1:length(hyposCardLocalD)
%                             hD = hyposLocalD(begsLocalD(jj):endsLocalD(jj));
%                             if(all(ismember(hI,hD)) && isnan(foundInDRelH(ii))) % All tracks in hI are also in hD
%                                 foundInDRelH(ii) = jj;
%                             end                               
%                         end
%                         for jj=1:length(hyposCardLocalDS)
%                             hDS = hyposLocalDS(begsLocalDS(jj):endsLocalDS(jj));
%                             if(all(ismember(hI,hDS)) && isnan(foundInDSRelH(ii))) % All tracks in hI are also in hD
%                                 foundInDSRelH(ii) = jj;
%                             end                               
%                         end
%                         for jj=1:length(hyposCardLocalP)
%                             hP = hyposLocalP(begsLocalP(jj):endsLocalP(jj));
%                             if(all(ismember(hI,hP)) && isnan(foundInPRelH(ii))) % All tracks in hI are also in hD
%                                 foundInPRelH(ii) = jj;
%                             end                               
%                         end                        
%                         for jj=1:length(hyposCardLocalPS)
%                             hPS = hyposLocalPS(begsLocalPS(jj):endsLocalPS(jj));
%                             if(all(ismember(hI,hPS)) && isnan(foundInPSRelH(ii))) % All tracks in hI are also in hD
%                                 foundInPSRelH(ii) = jj;
%                             end                               
%                         end                           
%                     end  
%                     
%                     probabilitiesH = exp(probLogLocal);
%                     probabilitiesH = probabilitiesH/sum(probabilitiesH); 
%                     maxProbH = max(probabilitiesH);
%                     
%                     probsHNotD = probabilitiesH(isnan(foundInDRelH));
%                     probsHNotDS = probabilitiesH(isnan(foundInDSRelH));
%                     probsHNotP = probabilitiesH(isnan(foundInPRelH));
%                     probsHNotPS = probabilitiesH(isnan(foundInPSRelH));
%                     
%                     if(~isempty(probsHNotD))
%                         pRatioD = maxProbH/max(probsHNotD);
%                     else
%                         pRatioD = Inf; 
%                     end
%                     if(~isempty(probsHNotDS))
%                         pRatioDS = maxProbH/max(probsHNotDS);
%                     else
%                         pRatioDS = Inf; 
%                     end
%                     if(~isempty(probsHNotP))
%                         pRatioP = maxProbH/max(probsHNotP);
%                     else
%                         pRatioP = Inf; 
%                     end              
%                     
%                     % Ratio of how many of the hypothesis of B&B are not in the other approaches
%                     
%                     dCountRatioH = sum(isnan(foundInDRelH))/length(hyposCardLocal);
%                     dSCountRatioH = sum(isnan(foundInDSRelH))/length(hyposCardLocal);
%                     pCountRatioH = sum(isnan(foundInPRelH))/length(hyposCardLocal);
%                     
%                     exploreAnalysis(end+1) =  struct('nHypoTotalMax',nHypoTotalMax,'nCap',nCap,'mCap',mCap,...
%                         'hypoCountM',size(probLogLocalM,2),'hypoCountH',size(probLogLocal,2),'hypoCountD',size(probLogLocalD,2),'hypoCountDS',size(probLogLocalDS,2),...
%                         'hypoCountP',size(probLogLocalP,2),'hypoCountPS',size(probLogLocalPS,2),'nCL',nCLD,'priorAveSize',mean(pcSizeD),'pqLen',pqLen,...
%                         'pMassMNotH',pMassMNotH,'pMassMNotD',pMassMNotD,'pMassMNotDS',pMassMNotDS,...
%                         'pMassMNotP',pMassMNotP,'pMassMNotPS',pMassMNotPS,...
%                         'pRatioD',pRatioD,'pRatioDS',pRatioDS,'pRatioP',pRatioP,...
%                         'dCountRatioH',dCountRatioH,'dSCountRatioH',dSCountRatioH,'pCountRatioH',pCountRatioH);
                    
%                     preExplore(end+1) = struct('hypos',hypos,'hyposCard',hyposCard,'clusters',clusters,'clustersCard',clustersCard,...
%                     'probLogHypos',probLogHypos,'iC',iC,'assocLocal',assocLocal,'gainMatPostC',gainMatPostC,...
%                         'indicesOfNewbornTracks',indicesOfNewbornTracks,'nHypoTotalMax',nHypoTotalMax,'trackNumberLookup',trackNumberLookup,'k',k);
                    
                    
%                     if(nHDS == length(hyposCardLocal) && nHDS < length(hyposCardLocalD))
%                         
%                        error('check here'); 
%                     end
                    
%                         exploreAnalysis = struct('nHypoTotalMax',{},'nCap',{},'mCap',{},'hypoCountH',{},'hypoCountD',{},...
%         'hypoCountP',{},'nCL',{},'priorAveSize',{},'pqLen',{},...
%         'pMassMNotH',{},'pMassMNotD',{},'pMassMNotDS',{},'pMassMNotP',{},'pMassMNotPS',{});
                    
                    
                    
%                     % Old exploration analysis
%                     
%                     % Should I for the sake of fairness unly use the topmost 150 hypotheses for D and P?
%                     
%                     [probLogSortedP,ix] = sort(probLogLocal2,'descend');
%                     nHP = min(length(ix),length(hyposCardLocal));
%                     probLogLocalP = probLogSortedP(1:nHP);
%                     hyposCardLocalP = hyposCardLocal2(ix(1:nHP));
%                     [aRemove,tCloudRemove] = pickIndC(ix(1:nHP),hyposCardLocal2);
%                     hyposLocalP = hyposLocal2(aRemove);
%                     
%                     
%                     % Can I do some comparison immediately here?
%                     
%                     foundInDouble = NaN*zeros(1,length(probLogLocal));
%                     foundInPropto = NaN*zeros(1,length(probLogLocal));
%                     foundInBranchbound = NaN*zeros(1,length(probLogLocal1));
%                     foundInBranchbound2 = NaN*zeros(1,length(probLogLocalP));
%                     
%                     begsLocalH = tCloud2BegInd(hyposCardLocal);
%                     endsLocalH = tCloud2EndInd(hyposCardLocal);
%                     begsLocalD = tCloud2BegInd(hyposCardLocal1);
%                     endsLocalD = tCloud2EndInd(hyposCardLocal1);    
%                     begsLocalP = tCloud2BegInd(hyposCardLocalP);
%                     endsLocalP = tCloud2EndInd(hyposCardLocalP);                        
%                     
%                     
%                     
%                     for ii=1:length(hyposCardLocal)
%                        
%                         hI = hyposLocal(begsLocalH(ii):endsLocalH(ii));
%                         
%                         % Is this hypothesis also in hyposLocal1?
%                         
%                         for jj=1:length(hyposCardLocal1)
%                            
%                             hD = hyposLocal1(begsLocalD(jj):endsLocalD(jj));
%                             
%                             if(all(ismember(hI,hD)) && isnan(foundInDouble(ii))) % All tracks in hI are also in hD
%                                 foundInDouble(ii) = jj;
%                             end
%                             if(all(ismember(hD,hI)) && isnan(foundInBranchbound(jj))) % All tracks in hI are also in hD
%                                 foundInBranchbound(jj) = ii;
%                             end                            
%                             
%                         end
%                     end
%                     
%                     for ii=1:length(hyposCardLocal)
%                        
%                         hI = hyposLocal(begsLocalH(ii):endsLocalH(ii));
%                         
%                         % Is this hypothesis also in hyposLocal1?
%                         
%                         for jj=1:length(hyposCardLocalP)
%                            
%                             hP = hyposLocalP(begsLocalP(jj):endsLocalP(jj));
%                             
%                             if(all(ismember(hI,hP)) && isnan(foundInPropto(ii))) % All tracks in hI are also in hD
%                                 foundInPropto(ii) = jj;
%                             end
%                             if(all(ismember(hP,hI)) && isnan(foundInBranchbound2(jj))) % All tracks in hI are also in hD
%                                 foundInBranchbound2(jj) = ii;
%                             end                            
%                             
%                         end
%                     end                    
%                     
%                     scoresHNotD = probLogLocal(isnan(foundInDouble));
%                     scoresHNotP = probLogLocal(isnan(foundInPropto));
%                     scoresDNotH = probLogLocal1(isnan(foundInBranchbound));
%                     scoresPNotH = probLogLocalP(isnan(foundInBranchbound2));
%                     
%                     scoreMedianDiffD = median(scoresHNotD) - median(scoresDNotH);
%                     scoreMedianRatioD = median(scoresHNotD)/median(scoresDNotH);
%                     scoreMedianDiffP = median(scoresHNotP) - median(scoresPNotH);
%                     scoreMedianRatioP = median(scoresHNotP)/median(scoresPNotH);                    
%                     
%                     probsH = exp(probLogLocal);
%                     probsH = probsH/sum(probsH);
%                     
%                     probsD = exp(probLogLocal1);
%                     probsD = probsD/sum(probsD);
%                     
%                     probsP = exp(probLogLocalP);
%                     probsP = probsP/sum(probsP);                    
%                     
% %                     exploreAnalysis(end+1) = struct('probMassHInD',sum(probsH(~isnan(foundInDouble))),...
% %                         'probMassDInH',sum(probsD(~isnan(foundInBranchbound))),...
% %                         'probMassHInP',sum(probsH(~isnan(foundInPropto))),...
% %                         'probMassPInH',sum(probsP(~isnan(foundInBranchbound2))),...
% %                         'nHypoTotalMax',nHypoTotalMax,'nCap',nCap,'mCap',mCap,'hypoCountH',size(probLogLocal,2),...
% %                         'hypoCountD',size(probLogLocal1,2),'hypoCountP',size(probLogLocal2,2),'nCL',nCL,'priorAveSize',mean(pcSize),'pqLen',pqLen,...
% %                         'scoreDMedianDiff',scoreMedianDiffD,'scoreDMedianRatio',scoreMedianRatioD,'scorePMedianDiff',scoreMedianDiffP,'scorePMedianRatio',scoreMedianRatioP);                    
%                     
%                     
% %                     if(scoreMedianDiffP < -0.8)
% %                        error('stio here'); 
% %                     end
% %                     if(mean(hyposCardLocal) > 1 && length(hyposCardLocal) > 14 && ~isempty(scoresHNotP) && k>4)
% %                        error('look at retu'); 
% %                     end
%                     
%                     
% %                     hyposCardLocal1
% %                     
% %                     if(size(hyposCardLocal1,2) > 16 && k>=20)
% %                        error('kkk'); 
% %                     end
                    
                    
                end

                
                t2BB = clock;
                
                begsH = tCloud2BegInd(hyposCardLocal);
                endsH = tCloud2EndInd(hyposCardLocal);
                
                newHypos = [newHypos,hyposLocal];
                newHyposCard = [newHyposCard,hyposCardLocal];
                newProbLogs = [newProbLogs,probLogLocal];
                
                
                newHyposOld = newHyposCount;
                newHyposCount = newHyposCount + size(hyposCardLocal,2);
                cWNew = [cWNew,(newHyposOld+1):newHyposCount];
                cCardWNew = [cCardWNew,size((newHyposOld+1):newHyposCount,2)];
                clusterNumber = iC;
                tracksInCluster = unique(hyposLocal);
                tfClusterMembership(tracksInCluster) = clusterNumber;
                
                
                if(k==9 && iC==2)
                    %error('Is Mecbi')
                end
                
                [shareMeasurements,shareTracks,shareClusters] = testClustersShareMeasurements(cWNew,cCardWNew,newHypos,newHyposCard,meaHistColNew);
                if(~isempty(shareMeasurements))
                    error('Measurements shared between clusters after double Murty 0');
                end
                
                
            end
            
            [boolsH,boolsT] = testRepeatedMeasurements(newHypos,newHyposCard,meaHistColNew);
            if(any(boolsH))
                
                [hyposNewB,hyposCardNewB,clustersNewB,clustersCardNewB,probabilitiesB,probLogsB] = sortHyposInCluster(newHypos,newHyposCard,cWNew,cCardWNew,newProbLogs);
                error('Did I get repeated measurements in a single hypothesis after main loop of DM?');
            end
            
            
            
            
            % Loop through all measurements to check if newborn track must be added as separate hypothesis
            
            for jj=1:m
                tIndex =indicesOfNewbornTracks(jj);  % Number of current track
                
                % Is this track a member of any hypothesis?
                % If not, then we need to add it as a separate hypothesis
                
                %if(~ismember(tIndex,newHypos))
                if(~ismember(jj,meaHistColNew(end,newHypos))) % New criterion only allows separate newborn cluster if MEASUREMENT not claimed in other clusters
                    hI = tIndex;
                    newHyposCard = [newHyposCard,1];
                    newHypos = [newHypos,hI];
                    newProbLogs = [newProbLogs,log(1)];
                    hypoNumber = size(newHyposCard,2);
                    tCluster = tfClusterMembership(tIndex);
                    if(isnan(tCluster))
                        cWNew = [cWNew,length(newHyposCard)];
                        cCardWNew = [cCardWNew,1];
                    else
                        [cWNew,cCardWNew] = insertElements(tCluster,hypoNumber,cWNew,cCardWNew,1);
                    end
                end
            end
            
            maxHyposCountMC(k,iMC,dd) = max(maxHyposCountMC(k,iMC,dd),size(newHyposCard,2));
            
            [hyposNewB,hyposCardNewB,clustersNewB,clustersCardNewB,probabilitiesB,probLogsB] = sortHyposInCluster(newHypos,newHyposCard,cWNew,cCardWNew,newProbLogs);
            
                    if(k==801)
                        
                        %error('look at hyposLocal before making pq-tree');
                    end            
            
            
            [shareMeasurements,shareTracks,shareClusters] = testClustersShareMeasurements(cWNew,cCardWNew,newHypos,newHyposCard,meaHistColNew);
            if(~isempty(shareMeasurements))
                error('Measurements shared between clusters after double Murty 1');
            end
            
            
            if(k==9)
                
                %error('look at Mechi PHD tracks');
            end
            
            
            if(k==1335)
                %error('look at Kaori');
            end
            
            
            if(k>=2)
                
                gainMatEltsVectorizedIndices = find(~isnan(trackNumberLookup))';
                
                
                x = v2m(gainMatEltsVectorizedIndices,size(trackNumberLookup,1));
                y = v2m(find(~isinf(gainMatFull))',size(trackNumberLookup,1));
                
                tracksEntire = trackNumberLookup(gainMatEltsVectorizedIndices);
                
                tracksActive = unique(newHypos);
                boolNotMissing = ismember(tracksEntire,tracksActive);
                missingTracksIx = find(~boolNotMissing);
                
                
                parentsMissing = zeros(size(missingTracksIx));
                meaMissing = zeros(size(missingTracksIx));
                tracksMissing = zeros(size(missingTracksIx));
                for ii=1:length(missingTracksIx)
                    tracksMissing(ii) = tracksEntire(missingTracksIx(ii));
                    parentsMissing(ii) = x(1,missingTracksIx(ii));
                    meaMissing(ii) = x(2,missingTracksIx(ii));
                    
                end
                
                if(doPostDMRecycle)
                    
                    ttpMissing = NaN*zeros(size(missingTracksIx));
                    phdTracksMissing = NaN*zeros(size(trackFile,1),length(missingTracksIx));
                    
                    
                    for ii=1:length(parentsMissing)
                        parent = parentsMissing(ii);
                        mea = meaMissing(ii);
                        if(parent <= size(trackFile,2) && mea <=m) %PHD update with parent contribution from MBM
                            
                            % Kinematic pdf has already been found in KF upate
                            
                            phdTracksMissing(inCol.tarX,ii) = trackFileNew(inCol.tarX,tracksMissing(ii));
                            phdTracksMissing(inCol.tarP,ii) = trackFileNew(inCol.tarP,tracksMissing(ii));
                            
                            % It remains to calculate the weight of this PHD component
                            
                            ttp = trackTotalProbs(parent);
                            ttpMissing(ii) = ttp;
                            if(ttp > 0.094)
                                %error('Check why I got such a large ttp value');
                            end
                            exi = trackFile(inCol.exi,parent);
                            numer = exp(gainMatFull(parent,mea))*ttp;
                            phdTracksMissing(inCol.exi,ii) = numer/(lambdaFa*exi + numer);
                            
                            
                            if(ii==52)
                                
                                %error('ew');
                            end
                            
                            
                        elseif(parent <= size(trackFile,2))    % Recyling a misdetected association
                            
                            % Again, kinematic pdf has already been found in KF upate
                            
                            phdTracksMissing(inCol.tarX,ii) = trackFileNew(inCol.tarX,tracksMissing(ii));
                            phdTracksMissing(inCol.tarP,ii) = trackFileNew(inCol.tarP,tracksMissing(ii));
                            
                            % It remains to calculate the weight of this PHD component
                            
                            ttp = trackTotalProbs(parent);
                            ttpMissing(ii) = ttp;
                            phdTracksMissing(inCol.exi,ii) = ttp*(1-pD);
                            
                            
                            
                            
                            if(tracksMissing(ii)==336 && k==9)
                                
                                %error('found the mechi track in PHD?');
                            end
                            
                        else   % PHD update with parent contribution from predicted PHD
                            
                            
                            
                            error('stop ere instead');
                            
                        end
                        
                    end
                    phdTracks = [phdTracks,phdTracksMissing];
                end
                %ttpMissing = trackTotalProbs(tracksMissing);
                %stateMissing = trackFile(inCol.tarX,tracksMissing);
                %covMissing = trackFile(inCol.tarP,tracksMissing);
                
                
                if(size(x,2) ~= size(y,2))
                    %error('I have a problem');
                end
                
                if(k==10)
                    %error('k at 10');
                end
                
                
            end
            
            
            
            hyposOld = hypos;
            hyposCardOld = hyposCard;
            clustersOld = clusters;
            clustersCardOld = clustersCard;
            probLogHyposOld = probLogHypos;
            meaHistColOld = meaHistCol;
            trackFileOld = trackFile;
            
            if(k==1335)
                %error('see what happens right after DM at 1335');
            end
            
            
            
            hypos = newHypos;
            hyposCard = newHyposCard;
            clusters = cWNew;
            clustersCard = cCardWNew;
            probLogHypos = newProbLogs;
            
            trackFile = trackFileNew;
            meaHistCol = meaHistColNew;
            trackFileShadow = trackFileShadowNew;
            
            
            [boolsH,boolsT] = testRepeatedMeasurements(hypos,hyposCard,meaHistCol);
            if(any(boolsH))
                error('Did I get repeated measurements in a single hypothesis after DM?');
            end
            
            % -------------------------------------------------------------
            % Done with Double Murty
            % -------------------------------------------------------------
            
            t2 = clock;
            deltaT = etime(t2,t1);
            timesArr(6,iMC,dd) = timesArr(6,iMC,dd) + deltaT;
            timesArrExt(6,k,iMC,dd) = deltaT;
            
            timesDmc(5,k,iMC,dd) = toc;
            tic
            
            [shareTracks,shareClusters] = testClustersShareTracks(cWNew,cCardWNew,newHypos,newHyposCard);
            if(~isempty(shareTracks))
                error('tracks shared among clusters before pruning a');
            end
            
            hyposCountStagesMC(2,k,iMC,dd) = length(newHyposCard);
            
            [hyposNewB,hyposCardNewB,clustersNewB,clustersCardNewB,probabilitiesB,probLogsB] = sortHyposInCluster(newHypos,newHyposCard,cWNew,cCardWNew,newProbLogs);
            begsH = tCloud2BegInd(hyposCardNewB);
            endsH = tCloud2EndInd(hyposCardNewB);
            
            [shareMeasurements,shareTracks,shareClusters] = testClustersShareMeasurements(clustersNewB,clustersCardNewB,hyposNewB,hyposCardNewB,meaHistColNew);
            if(~isempty(shareMeasurements))
                error('Measurements shared between clusters after double Murty');
            end
            
            
            [trackSumProbs,trackTotalProbs] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
            
            
            %             seg = [3,1,0,0]';
            %             k1 = 2;k2 = 5;kEval = k;
            %             [ttp,tracksCol,hyposCol,clustersCol,clustersAndTTP] = ttpSegment(seg,k1,k2,kEval,meaHistCol,trackFile,hypos,hyposCard,clusters,clustersCard,probLogHypos,inCol);
            
            [trackSumProbs,trackTotalProbs] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
            
            
            % -------------------------------------------------------------
            % Pruning -----------------------------------------------------
            % (With above modification mainly reduced to track pruning) ---
            % -------------------------------------------------------------
            
            
            
            t1 = clock;
            
            [shareTracks,shareClusters] = testClustersShareTracks(clusters,clustersCard,hypos,hyposCard);
            if(~isempty(shareTracks))
                error('tracks shared among clusters before pruning');
            end
            
            [hyposNewB,hyposCardNewB,clustersNewB,clustersCardNewB,probabilitiesB,probLogsB] = sortHyposInCluster(hypos,hyposCard,clusters,clustersCard,probLogHypos);
            
            
            
            if(~isempty(clusters)) % Only do pruning if we have more than 0 hypotheses
                designated = 1;
                
                [hypos,hyposCard,clustersNew,clustersNewCard,probLogHypos,trackFile,trackFileShadow,meaHistCol,existences] = ...
                    pruningPmbmBid(hypos,hyposCard,clusters,clustersCard,probLogHypos,trackFile,trackFileShadow,meaHistCol,inCol,nHypoTotalMax,k);
                
                
                hyposCount(k) = length(hyposCard);
                nTracks = size(trackFile,2);
                hypos = hypos;
                hyposCard = hyposCard;
                clusters = clustersNew;
                clustersCard = clustersNewCard;
                probLogHypos = probLogHypos;
                
                [probs,aSel,b,c] = probLogs2ProbabilitiesForHypos(clusters,probLogHypos,clusters,clustersCard,size(hyposCard,2));
                if(any(probs == 0))
                    error('Why do I have zero probability hypotheses?');
                end
                
            end
            mHS = size(meaHistCol);
            
            t2 = clock;
            deltaT = etime(t2,t1);
            timesArr(7,iMC,dd) = timesArr(7,iMC,dd) + deltaT;
            timesArrExt(7,k,iMC,dd) = deltaT;
            
            timesDmc(6,k,iMC,dd) = toc;
            tic
            
            maxHyposCountMC(k,iMC,dd) = max(maxHyposCountMC(k,iMC,dd),size(hyposCard,2));
            
            
            [shareTracks,shareClusters] = testClustersShareTracks(clusters,clustersCard,hypos,hyposCard);
            if(~isempty(shareTracks))
                error('tracks shared among clusters after pruning');
            end
            
            
            [shareMeasurements,shareTracks,shareClusters] = testClustersShareMeasurements(clusters,clustersCard,hypos,hyposCard,meaHistCol);
            if(~isempty(shareMeasurements))
                error('Measurements shared between clusters after pruning');
            end
            
            %[trackSumProbs,trackTotalProbs] = trackProbAccumulate(meaHistCol,trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
            [trackSumProbs,trackTotalProbs] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
            
            hyposCountStagesMC(3,k,iMC,dd) = length(hyposCard);
            [hyposNewA,hyposCardNewA,clustersNewA,clustersCardNewA,probabilitiesA,probLogsNewA] = sortHyposInCluster(hypos,hyposCard,clusters,clustersCard,probLogHypos);
            
            
            % -------------------------------------------------------------
            % N-scan track merging ----------------------------------------
            % -------------------------------------------------------------
            
            
            [hyposNewB,hyposCardNewB,clustersNewB,clustersCardNewB,probabilitiesB,probLogsB] = sortHyposInCluster(hypos,hyposCard,clusters,clustersCard,probLogHypos);
            
            
            
            [hypos,hyposCard,clusters,clustersCard,probLogHypos,meaHistCol,trackFile,trackFileShadow] ...
                = nScanMerge(hypos,hyposCard,clusters,clustersCard,probLogHypos,meaHistCol,inCol,trackFile,trackFileShadow);
            %end
            
            %hyposCard
            
            
            [hyposNewB,hyposCardNewB,clustersNewB,clustersCardNewB,probabilitiesB,probLogsB] = sortHyposInCluster(hypos,hyposCard,clusters,clustersCard,probLogHypos);
            
            
            [boolsH,boolsT] = testRepeatedMeasurements(hypos,hyposCard,meaHistCol);
            if(any(boolsH))
                error('Did I get repeated measurements in a single hypothesis after NSM?');
            end            
            
            
            
            % -------------------------------------------------------------
            % Cluster splitting ------------------------------------------
            % -------------------------------------------------------------
            
            t1 = clock;

            
            mahaCSThres = 3;
            [hyposNew,hyposNewCard,clustersNew,clustersNewCard,probabilitiesAS,splittingTracks] = clusterSplittingMajTraj(hypos,hyposCard,clusters,clustersCard,probLogHypos,trackFile,sig,meaHistCol,inCol,splittingThreshold,mahaCSThres,nonSplitLag,k);
            

            [hyposNewB,hyposCardNewB,clustersNewB,clustersCardNewB,probabilitiesB,probLogsB] = sortHyposInCluster(hyposNew,hyposNewCard,clustersNew,clustersNewCard,log(probabilitiesAS));

            t2 = clock;
            deltaT = etime(t2,t1);
            timesArr(8,iMC,dd) = timesArr(8,iMC,dd) + deltaT;
            timesArrExt(8,k,iMC,dd) = deltaT;
            
            
            timesDmc(7,k,iMC,dd) = toc;
            tic
            
            
            %             if(k==468)
            %                error('Do I have repeated tracks at the end of cycle 468 too? If so, why not eliminated by nScanMerge?');
            %             end
            
            
            maxHyposCountMC(k,iMC,dd) = max(maxHyposCountMC(k,iMC,dd),size(hyposCard,2));
            
            % -----------------------------------------------------------------
            % Recyling begins -------------------------------------------------
            % -----------------------------------------------------------------
            
            t1 = clock;
            
            skippedTracks = setdiff(hypos,hyposNew);
            if(size(skippedTracks,2) ~= size(splittingTracks,2))
                error('The skipped tracks should be equal to the splitting tracks');
            end
            nS = size(skippedTracks,2);
            
            [clusterNumbers,hyposCol,probabilities] = track2Cluster(skippedTracks,hypos,hyposCard,clusters,clustersCard,probLogHypos);
            wS = trackFile(inCol.exi,skippedTracks).*probabilities;
            
            phdTracksRecycle = trackFile(:,skippedTracks);
            phdTracksRecycle(inCol.exi,:) = wS;
            phdTracksRecycle(inCol.cost,:) = NaN;
            phdTracksRecycle(inCol.contrib,:) = NaN;
            phdTracksRecycle(inCol.visi,:) = NaN;
            phdTracksRecycle(inCol.label,:) = NaN;
            phdTracksRecycle(inCol.last,:) = NaN;
            
            phdTracks = [phdTracks,phdTracksRecycle];
            
            % Then I may also do some mixture reduction on Poisson component
            
            % Rduce the number of tracks in Poisson component
            
            if(doRunnall)
                wRunn = phdTracks(inCol.exi,:)';
                xRunn = phdTracks(inCol.tarX,:)';
                pRunn = permute(covVec2Mat(phdTracks(inCol.tarP,:)),[3,1,2]);
                mRunn = min(mRunnMax,max(ceil(size(wRunn,1)/4),2));
                [wreduced, xreduced, Preduced, costs, merges] = Runnalls(wRunn, xRunn, pRunn, mRunn);
                xPostRunn = xreduced';
                pPostRunn = permute(Preduced,[3,2,1]);
                wPostRunn = wreduced';
                phdTracksMR = NaN*zeros(size(phdTracks,1),size(wPostRunn,2));
                phdTracksMR(inCol.tarX,:) = xPostRunn;
                phdTracksMR(inCol.tarP,:) = covMat2Vec(pPostRunn);
                phdTracksMR(inCol.exi,:) = wPostRunn;
                phdTracks = phdTracksMR;
            else
                % If I dont use Runnall, use simple pruning instead
                
                % Try to do nothing!
                
                
            end
            
            phdTracks(:,phdTracks(inCol.exi,:) < 1e-5) = [];
            
            nTPHD = size(phdTracks,2);
            
            % -----------------------------------------------------------------
            % Recyling ends ---------------------------------------------------
            % -----------------------------------------------------------------
            
            % Remove tracks that have been removed from hypotheses during splitting
            
            [hyposNew,trackFile,trackFileShadow,meaHistCol] = trackPruningPmbm(hyposNew,trackFile,trackFileShadow,meaHistCol,k);
            nTracks = size(meaHistCol,2);
            
            %         mHCC = meaHistCol;
            %         mHCC(isnan(mHCC)) = -1;
            %         [x,y,z] = unique(mHCC','rows');
            %         if(length(unique(z)) ~= size(meaHistCol,2))
            %            disp(['Non-unique tracks after post-recyling pruning at k=',num2str(k)]);
            %            error('Why repeated tracks?');
            %         end
            
            
            
            
            
            %[trackSumProbs,trackTotalProbs] = trackProbAccumulate(meaHistCol,trackFile,inCol,hyposNew,hyposNewCard,clustersNew,clustersNewCard,log(probabilitiesAS));
            [trackSumProbs,trackTotalProbs] = trackProbAccumulatePure(trackFile,inCol,hyposNew,hyposNewCard,clustersNew,clustersNewCard,log(probabilitiesAS));
            probLogHypos = log(probabilitiesAS);
            hypos = hyposNew;
            hyposCard = hyposNewCard;
            clusters = clustersNew;
            clustersCard = clustersNewCard;
            trackFileOld = trackFile;
            
            
            
            if(k==9)
                %error('Is Mechi still on track here?');
            end
            
            [hyposNewA,hyposCardNewA,clustersNewA,clustersCardNewA,probabilitiesA,probLogsNewA] = sortHyposInCluster(hypos,hyposCard,clusters,clustersCard,probLogHypos);
            begsH = tCloud2BegInd(hyposCardNewA);
            endsH = tCloud2EndInd(hyposCardNewA);
            
            maxHyposCountMC(k,iMC,dd) = max(maxHyposCountMC(k,iMC,dd),size(hyposCardNewA,2));
            
            [boolsH,boolsT] = testRepeatedMeasurements(hypos,hyposCard,meaHistCol);
            if(any(boolsH))
                error('Did I get repeated measurements in a single hypothesis after CS?');
            end
            
            if(k==9)
                %error('stop');
            end
            
            %             if(k==3)
            %                 error('stop here');
            %             end
            
            
            if(k==612)
                %error('Are both Kaori tracks still in same cluster?');
            end
            
            
            t2 = clock;
            deltaT = etime(t2,t1);
            timesArr(9,iMC,dd) = timesArr(9,iMC,dd) + deltaT;
            timesArrExt(9,k,iMC,dd) = deltaT;
            
            
            if(k==611)
                %error('Look at Kartika - compare CM and Angel');
            end
            
            
            hyposCountStagesMC(4,k,iMC,dd) = length(hyposCard);
            
            if(doSuccessRates)
                
                timesDmc(8,k,iMC,dd) = toc;
                tic
                
                
                
                [shareTracks,shareClusters] = testClustersShareTracks(clustersNewA,clustersCardNewA,hyposNewA,hyposCardNewA);
                if(~isempty(shareTracks))
                    error('tracks shared among clusters after recycling');
                end
                
                
                
                
                % -----------------------------------------------------------------
                % Monitoring stuff ------------------------------------------------
                % -----------------------------------------------------------------
                
                % Check probability mass that includes true track
                t1 = clock;
                
                minKFromHT = max(1,k+1-maxLag);
                if(~isempty(hTrue))
                    meaSeq = hTrue(minKFromHT:k,1);
                    [tracks,hyposCol,clusterNumbers] = findTrack(meaSeq,meaHistCol,hypos,hyposCard,clusters,clustersCard);
                    
                    probabilitiesCell = probLogs2Probabilities(probLogHypos,clusters,clustersCard);
                    probabilities = zeros(1,length(hyposCard));
                    begsC = tCloud2BegInd(clustersCard);
                    endsC = tCloud2EndInd(clustersCard);
                    for iC=1:length(probabilitiesCell)
                        probabilities(clusters(begsC(iC):endsC(iC))) = probabilitiesCell{iC};
                    end
                    
                    if(~isempty(clusterNumbers))
                        
                        % Need to convert a-level hypothesis numbers to b-level
                        [b,c] = a2bcFaster(hyposCol{1},clustersCard);
                        
                        probabilitiesThis = probabilitiesCell{clusterNumbers};
                        pGood = sum(probabilitiesThis(b));
                    end
                end
                
                % Lets try to accumulate probability for each track in track file
                
                
                
                %[trackSumProbs,trackTotalProbs] = trackProbAccumulate(meaHistCol,trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
                [trackSumProbs,trackTotalProbs] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
                
                
                
                
                % -----------------------------------------------------------------
                % Success rate tests ----------------------------------------------
                % -----------------------------------------------------------------
                
%                 duplicatesFound = checkDuplicates(hypos,hyposCard,clusters,clustersCard);
%                 if(duplicatesFound)
%                     error('Found duplicate hypotheses');
%                 end
                
                
                [hyposS,hyposCardS,clustersS,clustersCardS,probsS] = sortHyposInCluster(hypos,hyposCard,clusters,clustersCard,probLogHypos);
                
                [shareTracks,shareClusters] = testClustersShareTracks(clustersS,clustersCardS,hyposS,hyposCardS);
                if(~isempty(shareTracks))
                    error('tracks shared among clusters after entire cycle');
                end
                
                [shareMeasurements,shareTracks,shareClusters] = testClustersShareMeasurements(clustersS,clustersCardS,hyposS,hyposCardS,meaHistCol);
                if(~isempty(shareMeasurements))
                    error('Measurements shared between clusters after entire cycle');
                end
                
                % Cardinality distribution for each cluster
                
                if(ismember(k,kInvestigate))
                    
                    cBegs = tCloud2BegInd(clustersCardS);
                    cEnds = tCloud2EndInd(clustersCardS);
                    
                    hBegs = tCloud2BegInd(hyposCardS);
                    hEnds = tCloud2EndInd(hyposCardS);
                    
                    cardAccum = zeros(size(clustersCardS,2)+1,maxCard+1);
                    
                    % First element horizontally is cardinality zero
                    % Last element vertically is Poisson component
                    
                    for cc=1:size(clustersCardS,2)
                        
                        inC = clustersS(cBegs(cc):cEnds(cc));
                        probsC = probsS(cBegs(cc):cEnds(cc));
                        
                        hyposCountPerCluster(cc,iMC,k,dd) = size(probsC,2);
                        
                        maxHypoSizePerCluster(cc,iMC,k,dd) = max(hyposCardS(cBegs(cc):cEnds(cc)));
                        
                        for ii=1:size(probsC,2)
                            h = hyposS(hBegs(inC(ii)):hEnds(inC(ii)));
                            exiH = trackFile(inCol.exi,h);
                            n = size(exiH,2);
                            if(n > 0)
                                combinations = dec2bin(2^n-1:-1:0)-'0';
                                
                                combiProbs = zeros(1,size(combinations,1));
                                
                                exiProds = zeros(1,size(combinations,1));
                                for jj=1:size(combinations,1)
                                    com = combinations(jj,:);
                                    exiProd = prod(exiH.^com)*prod((1-exiH).^(1-com));  % Thus this even make sense? I
                                    exiProds(jj) = exiProd;
                                    
                                    cardinalityJ = sum(com);
                                    cardAccumOld = cardAccum;
                                    cardAccum(cc,cardinalityJ+1) = cardAccum(cc,cardinalityJ+1) + exiProd*probsC(ii);
                                    combiProbs(jj) = exiProd;
                                end
                            else
                                cardAccum(cc,1) = cardAccum(cc,1) + probsC(ii);
                                combiProbs = 1;
                            end
                            
                            if(k==10)
                                if(sum(combiProbs ) < 0.999999999)
                                    error('what went wrong');
                                end
                            end
                        end
                    end
                    
                    % Cardinality contributions from Poisson component
                    
                    nAvePoiss = sum(phdTracks(inCol.exi,:));
                    cardAccum(end,:) = poisspdf(0:maxCard,nAvePoiss);
                    ca = cardAccum';
                    
                    phdAveCardMC(k,iMC,dd) = nAvePoiss;
                    
                    
                    
                    
                    
                    x = [1;zeros(maxCard,1)];
                    for c=1:size(ca,2)
                        x = conv(x,ca(:,c));
                        x = x(1:maxCard+1);
                    end
                    cardsProb = x;
                    cardsArr(:,k,iMC) = cardsProb';
                    
                    %        error('st');
                    
                    % Success rate stuff
                    
                    %                     [temp,ix] = max(cardsProb);
                    %                     mapCard = ix-1; % MAP cardinality estimate at this time / MC-run
                    %
                    %
                    %                     if(~isempty(hTrue))
                    %                         hTrueCopy = hTrue;
                    %                         hTrueCopy(isnan(hTrueCopy)) = -1;
                    %                         trueCard = sum(hTrueCopy(k,:) > -1);
                    %                     else
                    %                         trueCard = 0;
                    %                     end
                    
                    
                    hyposCountMC(k,iMC,dd) = length(hyposCard);
                    hyposProbMC(iMC,dd) = sum(vec(cardAccum(1:(end-1),2:end)));
                    if(hyposProbMC(iMC,dd) == 1)
                        %error('why?');
                    end
                    
                    
                    
                    
                    %successOrFailure = mapCard == trueCard;
                    
                    kIndex = find(kInvestigate == k);
                    [successOrFailure,mapCard,trueCard] = successEvaluateForCardinality(hTrue,cardsProb,k);
                    successOrFailureDmc(kIndex,iMC,dd) = successOrFailure;
                    pTrueDmc(kIndex,iMC,dd) = cardsProb(trueCard+1);
                    mapCardMC(k,iMC,dd) = mapCard;
                    
                    
                    if(doAngel)
                        
                        
                        cardsProbA = cardsArrAngel(:,k)';
                        [successOrFailureA,mapCardA,trueCardA] = successEvaluateForCardinality(hTrue,cardsProbA,k);
                        successOrFailureDmcAngel(kIndex,iMC,dd) = successOrFailureA;
                        
                        
                        cardTrue = sum(~isnan(hTrue(k,:)));
                        
                        
                        bestProb = max(cardsProbA);
                        trueProb = cardsProbA(cardTrue+1);
                        
                        cardConsistencyRatioAngel(kIndex,iMC,dd) = trueProb/bestProb;
                        
                        pBestAveAngel(kIndex,iMC,dd) = bestProb;
                        pTrueAveAngel(kIndex,iMC,dd) = trueProb;
                        
                        
                    end
                    
                    
                    
                end
                
                
                
                
                if(max(hypos) > size(trackFile,2))
                    error('hypos refers to non-existent tracks');
                end
                
                %                 cSize = size(clustersCard,2);
                %                 if(cSize > 0)
                %                    error('Finally got a non-empty cluster!');
                %                 end
                
                
                % GOSPA stuff
                
                %c_gospa=10; %Parameter c of the GOSPA metric. We also consider p=2 and alpha=2
                xEstGlobal = zeros(dimTar,0);
                for c=1:size(clustersCardS,2)
                    [hInC,tCloudRemove] = pickIndC(c,clustersCardS);
                    
                    % Now aRemove should be the hypothesis numbers within cluster c
                    
                    hyposCardC = hyposCardS(hInC);
                    probabilitiesC = probsS(hInC);
                    [hContentInC] = pickIndC(hInC,hyposCardS);
                    hyposC = hyposS(hContentInC);
                    xEst = estimator3(trackFile,hyposC,hyposCardC,probabilitiesC,inCol);
                    xEstGlobal = [xEstGlobal,xEst];
                end
                
                
                %[hyposA,hyposCardA,meaHistColA,trackWNumbers,trackHypoNumbers,ttpsA,probsA,trackFileA] = angelToEdmund(filter_upd,scenario(1,2).zCard,k,inCol);
                
                [temp,ttps] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);
                ttpsOfTrueTracks = trueTracksTTP(ttps,meaHistCol,hTrue,k);
                
                trueTrackTTPHistCM(k,1:size(ttpsOfTrueTracks,2),dd,iMC) = ttpsOfTrueTracks;
                
                
                
                % Then I should compare this with ground truth
                
                hTrueActive = find(~isnan(hTrue(k,:)));
                xTrueGlobal = zeros(dimTar,0);
                
                for tt=1:size(hTrueActive,2)
                    
                    targetT = scenario(iMC,dd).targetsTrue(hTrueActive(tt));
                    kInK = k-targetT.k(1)+1;
                    xT = targetT.x(:,kInK);
                    xTrueGlobal = [xTrueGlobal,xT];
                    
                end
                
                
                [d_gospa, ~, decomp_cost] = GOSPA(xTrueGlobal, xEstGlobal, 2, c_gospa, 2);
                
                squared_gospa=d_gospa^2;
                gospa_loc=decomp_cost.localisation;
                gospa_mis=decomp_cost.missed;
                gospa_fal=decomp_cost.false;
                
                squared_gospa_MC(k,iMC) = squared_gospa;
                squared_gospa_loc_MC(k,iMC) = gospa_loc;
                squared_gospa_false_MC(k,iMC)= gospa_fal;
                squared_gospa_mis_MC(k,iMC)= gospa_mis;
                
                
                % DMC stuff
                
                
                %cardConsistencyRatioDmc = zeros(length(kInvestigate),nMC,length(pDList));
                
                
                if(ismember(k,kInvestigate))
                    kIndex = find(kInvestigate == k);
                    
                    
                    %if(kIndex == 1)
                    
                    cardTrue = sum(~isnan(hTrue(k,:)));
                    
                    
                    bestProb = max(cardsProb);
                    trueProb = cardsProb(cardTrue+1);
                    cardConsistencyRatioDmc(kIndex,iMC,dd) = trueProb/bestProb;
                    
                    pBestAveDmc(kIndex,iMC,dd) = bestProb;
                    pTrueAveDmc(kIndex,iMC,dd) = trueProb;
                    
                    ospaDmc(kIndex,iMC,dd) = squared_gospa;
                    ospaLocDmc(kIndex,iMC,dd) = gospa_loc;
                    ospaFalseDmc(kIndex,iMC,dd) = gospa_fal;
                    ospaMisDmc(kIndex,iMC,dd) = gospa_mis;
                    
                    
                    %error('stoiph');
                    %end
                    
                    
                end
                
                if(k==200)
                    
                    %error('stop at 200 to look at GOSPA');
                end
                
                t2 = clock;
                deltaT = etime(t2,t1);
                timesArr(10,iMC,dd) = timesArr(10,iMC,dd) + deltaT;
                timesArrExt(10,k,iMC,dd) = deltaT;
                
                timesDmc(9,k,iMC,dd) = toc;
                
                
                
            end
            
            
%             if(k==610)
%                 error('stop here to look at Karina and Kartika');
%                 
%             end
            
            
            % -----------------------------------------------------------------
            % Plotting --------------------------------------------------------
            % -----------------------------------------------------------------
            
            
            if(doPlot && finalMC == 1)
                
                % Visualize PHD component
                
                phdGrid = muPPP*ones(size(phdGrid));
                
                for ii=1:nTPHD
                    muI = system.hMat*phdTracks(inCol.tarX,ii);
                    coI = system.hMat*covVec2Mat(phdTracks(inCol.tarP,ii))*system.hMat';
                    wI = phdTracks(inCol.exi,ii);
                    
                    % Can I vectorize the Gaussian evaluation?
                    
                    
                    vals = wI*exp(normpdfLog(pMonB,muI,coI));
                    phdGrid = phdGrid + reshape(vals,[nXB,nYB]);
                    
                end
                
                imagesc(xB,yB,log(phdGrid'));axis xy;
                
                %caxis([0,6e-3]);
                caxis([-11.5,-5]);
                hold on;
                
                % Visualize MBM component
                
                dimXVis = 1;
                dimYVis = 2;
                nXVis = 340;
                nYVis = 335;
                lims = [xMinDisp,xMaxDisp,yMinDisp,yMaxDisp];
                %             [gridX,gridY,gridVals] = mbm2phd(dimXVis,dimYVis,lims,nXVis,nYVis,clusters,clustersCard,hypos,hyposCard,probLogHypos,trackFile,inCol);
                %             imagesc(gridX,gridY,gridVals');axis xy;
                colorbar;
                
                nH = length(hyposCard);
                nC = length(clustersCard);
                
                begsC = tCloud2BegInd(clustersCard);
                endsC = tCloud2EndInd(clustersCard);
                
                begsH = tCloud2BegInd(hyposCard);
                endsH = tCloud2EndInd(hyposCard);
                
                
                
                
                nTracks = size(meaHistCol,2);
                for iT=1:nTracks
                    %showTrack(iT,k,meaHistCol,trackFileShadow,scenario(iMC,dd).zList,scenario(iMC,dd).zCard,inCol,colorArr(mod(iT-1,18)+1,:));
                    showTrack(iT,k,meaHistCol,trackFileShadow,scenario(iMC,dd).zList,scenario(iMC,dd).zCard,inCol,[1,1,1]);
                    hold on;
                end
                
                
                for iC=1:nC % Loop through all clusters
                    
                    hInC = clusters(:,begsC(iC):endsC(iC));
                    [probs,aSel,b,c] = probLogs2ProbabilitiesForHypos(hInC,probLogHypos,clusters,clustersCard,size(hyposCard,2));
                    
                    % Only display 10 topmost hypotheses from each cluster
                    
                    nHDisp = min(15,clustersCard(iC));
                    [pValsSorted,ix] = sort(probs,'descend');
                    probsSorted = probs(ix);
                    selHInC = hInC(ix(1:nHDisp)); % These should be bona-fide hypothesis numbers
                    for iH=1:length(selHInC)
                        tracksInH = hypos(:,begsH(selHInC(iH)):endsH(selHInC(iH)));
                        probsWithExi = probsSorted(iH)*trackFile(inCol.exi,tracksInH);
                        toBeDisplayed = find(probsWithExi > trackProbLimitDisp);
                        for ii=1:length(toBeDisplayed)
                            %hold on;
                            %showTrack(toBeDisplayed(ii),k,meaHistCol,trackFileShadow,scenario(iMC,dd).zList,scenario(iMC,dd).zCard,inCol,[1,1,1]);
                            hold on;
                            x = trackFile(inCol.tarX(1:2),tracksInH(toBeDisplayed(ii)));
                            pRaw = trackFile(inCol.tarP,tracksInH(toBeDisplayed(ii)));
                            pMat = covVec2Mat(pRaw);
                            pMat = pMat(1:2,1:2);
                            plot(x(1,:),x(2,:),'w*');
                            %plot(x(1,:),x(2,:),'*','color',colorsClusters(iC,:));
                            hold on;
                            el = getellipse03(x,pMat,1,100);
                            %el = getellipse03(x,pMat*nSigma^2,1,100);
                            plot(el(1,:),el(2,:),'w');
                            %plot(el(1,:),el(2,:),'color',colorsClusters(iC,:));
                            
                            
                            
                            
                        end
                        
                        
                    end
                end
                
                hold on;                
                for t=1:length(scenario(iMC,dd).targetsTrue)
                    kTList = scenario(iMC,dd).targetsTrue(t).k;
                    kI = find(scenario(iMC,dd).targetsTrue(t).k == k);
                    %plot(scenario(iMC,dd).targetsTrue(t).x(1,1:kI),scenario(iMC,dd).targetsTrue(t).x(2,1:kI),'g','linewidth',2);
                    %plot(scenario(iMC,dd).targetsTrue(t).x(1,1:kI),scenario(iMC,dd).targetsTrue(t).x(2,1:kI),'color',colorArr(t+1,:),'linewidth',2);
                    hold on;
                    
                    if(~isempty(kI))
                        sv = zeros(7,1);
                        sv(1:2) = scenario(iMC,dd).targetsTrue(t).x(1:2,kI);
                        yaw = atan2( scenario(iMC,dd).targetsTrue(t).x(4,kI),scenario(iMC,dd).targetsTrue(t).x(3,kI));
                        eul = [0,0,yaw]';
                        sv(4:7) = ypr2quat(eul);
                        plotShip(sv,colorArr(t+1,:));
                    end
                    hold on;
                end
                
                plot(measurements(1,:),measurements(2,:),'ro','linewidth',2);

                
%                 if(doAngel)
%                     
%                     if(ismember(k,kInvestigate))
%                         
%                         plot(xEstGlobal(1,:),xEstGlobal(2,:),'cd','linewidth',2);
%                     end
%                     
%                     xEst = reshape(X_estimate,[4,length(X_estimate)/4]);
%                     
%                     plot(xEst(1,:),xEst(2,:),'k+','linewidth',2);
%                     
%                 end
                
                
                
                
                hold off;
                axis([xMinDisp,xMaxDisp,yMinDisp,yMaxDisp]);
                %
                title(['Tracking results at k = ',num2str(k)]);
                xlabel('x [m]');
                ylabel('y [m]');
                pause(0.3);
                
                
                if(k==617)
                    
                   error('stop ere'); 
                end
                
                if(doMovie  )
                    %frame = getframe(fig1);
                    %writeMovie(movObj,frame);
                    
                    
                    frame = getframe(gcf);
                    writeVideo(v,frame);
                end
            end
            
            if(k==600 )
               %error('Store stuff at this time so I can initialize warm start with 400 hypotheses. I have to do that.');
            end
            %error('kkk');
            
        end
        if(doMovie && doPlot && iMC ==plotMC)
            %close(movObj);
            close(v);
        end
        %     randState
        %     hTrue
        %      sfa = successOrFailureArr(:,iMC)
        %      carr = cardsArr(:,:,iMC)
        %      stt = size(targetsTrue)
        
        
        %clear hyposNew clustersNew hypos;
        %        end
        
        
        
        if(mod(iMC,10) == 0 || iMC == nMC)
            
            
            % Save stuff in case simulation gets interrupted.
            
            savedID = [iMC,dd];
            save('../pmbm_large_files/tempFileCM11.mat');
            disp(['Saved stuff at iMC ',num2str(iMC),' dd ',num2str(dd)]);
            
        end
        
        
        
    end
    
    pTrueAve = mean(pTrueDmc(:,:,dd),2);
    pBestAve = mean(successOrFailureDmc(:,:,dd),2);
    [pBestAve,pTrueAve,pBestAve.^2]
    
    
    pTrueAve = mean(pTrueAveAngel(:,:,dd),2);
    pBestAve = mean(successOrFailureDmcAngel(:,:,dd),2);
    [pBestAve,pTrueAve,pBestAve.^2]
    
    iMCBeg = 1;
    
    

    
    %error('stop now');
end




% %[pBestAve > pTrueAve, pBestAve.^2 < pTrueAve]

%
% countTrueNonEmpty = 0;
% countBestNonEmpty = 0;
% for iMC=1:nMC
%     hTrue = hTrueColMC{1,iMC};
%     hBest = hTrueColMC{2,iMC};
%     if(~isempty(hTrue))
%         countTrueNonEmpty = countTrueNonEmpty + 1;
%     end
%     if(~isempty(hBest))
%         countBestNonEmpty = countBestNonEmpty + 1;
%     end
% end

