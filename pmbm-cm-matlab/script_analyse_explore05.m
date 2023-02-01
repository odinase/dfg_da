% exploreAnalysis = 
% 
%   1×4767 struct array with fields:
% 
%     nHypoTotalMax
%     nCap
%     mCap
%     hypoCountM
%     hypoCountH
%     hypoCountD
%     hypoCountDS
%     hypoCountP
%     hypoCountPS
%     nCL
%     priorAveSize
%     pqLen
%     pMassMNotH
%     pMassMNotD
%     pMassMNotDS
%     pMassMNotP
%     pMassMNotPS
%     pRatioD
%     pRatioDS
%     pRatioP
%     dCountRatioH
%     dSCountRatioH
%     pCountRatioH


% Make statistics over cardinalities and cluster sizes

maxAveCard = ceil(max(vertcat(exploreAnalysis.priorAveSize)));
maxNCL = max(vertcat(exploreAnalysis.nCL));

aveCardList = 1:maxAveCard;
ncList = 1:maxNCL;


aveLowLimList = [0,0.5,(1:14)+0.5];
aveHighLimList = [aveLowLimList(2:end),29];
countCaseMat = zeros(length(aveLowLimList),length(ncList));
for ii=1:length(aveLowLimList)
    for jj=1:length(ncList)
        
        ixIBool = vertcat(exploreAnalysis.priorAveSize) > aveLowLimList(ii) & vertcat(exploreAnalysis.priorAveSize) <= aveHighLimList(ii);
        ixJBool = vertcat(exploreAnalysis.nCL) == jj;
        countCaseMat(ii,jj) = sum(ixIBool & ixJBool);
    end
end

% Single-cluster analysis

aveLowLimListSC = [0,0.5,(1:7)+0.5];
aveHighLimListSC = [aveLowLimListSC(2:end),29];
countCaseMatSC = NaN*zeros(length(aveLowLimList),length(ncList));
for ii=1:length(aveLowLimListSC)

        
        ixIBool = vertcat(exploreAnalysis.priorAveSize) > aveLowLimListSC(ii) & vertcat(exploreAnalysis.priorAveSize) <= aveHighLimListSC(ii);
        ixJBool = vertcat(exploreAnalysis.nCL) > 1;
        countCaseMatSC(ii,1) = sum(ixIBool & ixJBool);

end


% Do quantile analysis for the pMass-ratios - Single-cluster

quanti = 0.90;
quantiPMassSC = zeros(length(aveLowLimListSC),4);
for cardi=1:length(aveLowLimListSC)
    
    ixIBool = vertcat(exploreAnalysis.priorAveSize) > aveLowLimListSC(cardi) & vertcat(exploreAnalysis.priorAveSize) <= aveHighLimListSC(cardi);
    ixJBool = vertcat(exploreAnalysis.nCL) > 1;
    ea = exploreAnalysis(ixIBool & ixJBool);
    
    
    quantiPMassSC(cardi,1) = quantile(vertcat(ea.pMassMNotH),quanti);
    quantiPMassSC(cardi,2) = quantile(vertcat(ea.pMassMNotD),quanti);
    quantiPMassSC(cardi,3) = quantile(vertcat(ea.pMassMNotDS),quanti);
    quantiPMassSC(cardi,4) = quantile(vertcat(ea.pMassMNotP),quanti);
    
end
figure;
plot(quantiPMassSC(:,1),'r');
hold on;
plot(quantiPMassSC(:,2),'b');
plot(quantiPMassSC(:,3),'g');
plot(quantiPMassSC(:,4),'k');
hold off;
legend('Branch-and-bound','Double','Double Reduced','Proportional');
xlabel('Average prior combination cardinality');
ylabel('90 percent quantile');
title('Frequency of cases: 10 percent probability lost');


pRatioBound = 10;
pRatioSmallSC = zeros(length(aveLowLimListSC),2);  % D  and P
for cardi=1:length(aveLowLimListSC)
    
    ixIBool = vertcat(exploreAnalysis.priorAveSize) > aveLowLimList(cardi) & vertcat(exploreAnalysis.priorAveSize) <= aveHighLimList(cardi);
    ixJBool = vertcat(exploreAnalysis.nCL) > 1;
    ea = exploreAnalysis(ixIBool & ixJBool);    
    
    pRatioSmallSC(cardi,1) = sum(horzcat(ea.pRatioD) < pRatioBound)/length(ea);
    %pRatioSmallSC(cardi,2) = sum(horzcat(ea.pRatioDS) < pRatioBound)/length(ea);
    pRatioSmallSC(cardi,2) = sum(horzcat(ea.pRatioP) < pRatioBound)/length(ea);
end


pCountMediansSC = zeros(length(aveLowLimListSC),2);  % D  and P
for cardi=1:length(aveLowLimListSC)
    
    ixIBool = vertcat(exploreAnalysis.priorAveSize) > aveLowLimList(cardi) & vertcat(exploreAnalysis.priorAveSize) <= aveHighLimList(cardi);
    ixJBool = vertcat(exploreAnalysis.nCL) > 1;

    
    
    
    ea = exploreAnalysis(ixIBool & ixJBool);
    pCountMediansSC(cardi,1) = median(horzcat(ea.dCountRatioH));
    %pCountMediansSC(cardi,2) = median(horzcat(ea.dSCountRatioH));
    pCountMediansSC(cardi,2) = median(horzcat(ea.pCountRatioH));
    
    
%     if(cardi == 3)
%        error('why so many cases here'); 
%     end    
    
end



error('pause here after SC');


% Do quantile analysis for the pMass-ratios

quanti = 0.90;

quantiPMassArr = zeros(length(aveLowLimList),4); % H, D, DS and P

for cardi=1:length(aveLowLimList)
    
    ixIBool = vertcat(exploreAnalysis.priorAveSize) > aveLowLimList(cardi) & vertcat(exploreAnalysis.priorAveSize) <= aveHighLimList(cardi);
    ixJBool = vertcat(exploreAnalysis.nCL) > 1;
    ea = exploreAnalysis(ixIBool & ixJBool);
    
    
    quantiPMassArr(cardi,1) = quantile(vertcat(ea.pMassMNotH),quanti);
    quantiPMassArr(cardi,2) = quantile(vertcat(ea.pMassMNotD),quanti);
    quantiPMassArr(cardi,3) = quantile(vertcat(ea.pMassMNotDS),quanti);
    quantiPMassArr(cardi,4) = quantile(vertcat(ea.pMassMNotP),quanti);
    
end






cutoffSize = 7; 
useNCL = [1:10];

ixBool1 = horzcat(exploreAnalysis.priorAveSize) > cutoffSize;
ixBool2 = ismember(horzcat(exploreAnalysis.nCL),useNCL);
ea = exploreAnalysis(ixBool1 & ixBool2);

% figure;histogram(horzcat(ea.dCountRatioH),100);title('dCountRatioH');
% figure;histogram(horzcat(ea.pCountRatioH),100);title('pCountRatioH');






figure;
plot(quantiPMassArr(:,1),'r');
hold on;
plot(quantiPMassArr(:,2),'b');
plot(quantiPMassArr(:,3),'g');
plot(quantiPMassArr(:,4),'k');
hold off;
legend('Branch-and-bound','Double','Double Reduced','Proportional');
xlabel('Average prior combination cardinality');
ylabel('90 percent quantile');
title('Probability mass that exploration method fails to account for');


% Use frequency of worst p-ratio being less than 5? (1.6094 in logarithm)?
% Again as a function of cardinality
% 
% figure;histogram(log(horzcat(ea.pRatioD)),100);title('pRatioD');
% figure;histogram(log(horzcat(ea.pRatioP)),100);title('pRatioP');

pRatioBound = 10;
pRatioSmall = zeros(length(aveLowLimList),3);  % D, DS and P


for cardi=1:length(aveLowLimList)
    
    ixIBool = vertcat(exploreAnalysis.priorAveSize) > aveLowLimList(cardi) & vertcat(exploreAnalysis.priorAveSize) <= aveHighLimList(cardi);
    ixJBool = vertcat(exploreAnalysis.nCL) > 1;
    ea = exploreAnalysis(ixIBool & ixJBool);    
    
    pRatioSmall(cardi,1) = sum(horzcat(ea.pRatioD) < pRatioBound)/length(ea);
    pRatioSmall(cardi,2) = sum(horzcat(ea.pRatioDS) < pRatioBound)/length(ea);
    pRatioSmall(cardi,3) = sum(horzcat(ea.pRatioP) < pRatioBound)/length(ea);
end

figure;
plot(pRatioSmall(:,1),'b');
hold on;
plot(pRatioSmall(:,2),'g');
plot(pRatioSmall(:,3),'k');
hold off;
legend('Double','Double Reduced','Proportional');
title('Frequency of cases where the best missed hypothesis is more probable than 1/10 of top hypothesis');
xlabel('Average prior combination cardinality');


pCountMedians = zeros(length(aveLowLimList),3);  % D, DS and P
for cardi=1:length(aveLowLimList)
    
    ixIBool = vertcat(exploreAnalysis.priorAveSize) > aveLowLimList(cardi) & vertcat(exploreAnalysis.priorAveSize) <= aveHighLimList(cardi);
    ixJBool = vertcat(exploreAnalysis.nCL) > 1;
    
    ea = exploreAnalysis(ixIBool & ixJBool);
    pCountMedians(cardi,1) = median(horzcat(ea.dCountRatioH));
    pCountMedians(cardi,2) = median(horzcat(ea.dSCountRatioH));
    pCountMedians(cardi,3) = median(horzcat(ea.pCountRatioH));
end

figure;
plot(pCountMedians(:,1),'b');
hold on;
plot(pCountMedians(:,2),'g');
plot(pCountMedians(:,3),'k');
hold off;
legend('Double','Double Reduced','Proportional');
title('Median ratio of missed hypotheses in Double and Proportional compared to Branch-and-bound');
xlabel('Average prior combination cardinality');

% Old stuff

% cutoffSize = 3; 
% useNCL = [1:10];
% 
% 
% ixBool2 = ismember(horzcat(exploreAnalysis.nCL),useNCL);
% 
% 
% ixBool1 = horzcat(exploreAnalysis.priorAveSize) > cutoffSize;
% ea = exploreAnalysis(ixBool1 & ixBool2);
% 
% 
% figure;
% histogram(horzcat(ea.probMassHInD),100);title('Prob H in D');
% 
% figure;
% histogram(horzcat(ea.probMassDInH),100);title('Prob D in H');
% 
% figure;
% histogram(horzcat(ea.probMassPInH),100);title('Prob P in H');
% 
% figure;
% histogram(horzcat(ea.probMassHInP),100);title('Prob H in P');
% 
% figure;hist(horzcat(ea.scoreDMedianDiff));
% title('Difference of median scores not in H and D');
% 
% 
% figure;hist(horzcat(ea.scorePMedianDiff));
% title('Difference of median scores not in H and P');