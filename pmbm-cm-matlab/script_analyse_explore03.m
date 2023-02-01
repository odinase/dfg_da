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

countCaseMat = zeros(length(aveCardList),length(ncList));
for ii=1:length(aveCardList)
    for jj=1:length(ncList)
        
        ixIBool = vertcat(exploreAnalysis.priorAveSize) > ii-1 & vertcat(exploreAnalysis.priorAveSize) <= ii;
        ixJBool = vertcat(exploreAnalysis.nCL) == jj;
        countCaseMat(ii,jj) = sum(ixIBool & ixJBool);
        
        
    end
end

% Do quantile analysis for the pMass-ratios

quanti = 0.80;

quantiPMassArr = zeros(length(aveCardList),4); % H, D, DS and P

for cardi=1:length(aveCardList)
    
    ixIBool = vertcat(exploreAnalysis.priorAveSize) > cardi-1 & vertcat(exploreAnalysis.priorAveSize) <= cardi;
    ea = exploreAnalysis(ixIBool);
    
    
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
ylabel('80 percent quantile');
title('Probability mass that exploration method fails to account for');


% Use frequency of worst p-ratio being less than 5? (1.6094 in logarithm)?
% Again as a function of cardinality
% 
% figure;histogram(log(horzcat(ea.pRatioD)),100);title('pRatioD');
% figure;histogram(log(horzcat(ea.pRatioP)),100);title('pRatioP');

pRatioBound = 10;
pRatioSmall = zeros(length(aveCardList),3);  % D, DS and P


for cardi=1:length(aveCardList)
    
    ixIBool = vertcat(exploreAnalysis.priorAveSize) > cardi-1 & vertcat(exploreAnalysis.priorAveSize) <= cardi;
    ea = exploreAnalysis(ixIBool);    
    
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


pCountMedians = zeros(length(aveCardList),3);  % D, DS and P
for cardi=1:length(aveCardList)
    
    ixIBool = vertcat(exploreAnalysis.priorAveSize) > cardi-1 & vertcat(exploreAnalysis.priorAveSize) <= cardi;
    ea = exploreAnalysis(ixIBool);
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