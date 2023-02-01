% exploreAnalysis = 
% 
%   1×21633 struct array with fields:
% 
%     probMassHInD
%     probMassDInH
%     nHypoTotalMax
%     nCap
%     mCap
%     hypoCountH
%     hypoCountD
%     nCL
%     priorAveSize

%load('../pmbm_large_files/tempFileCM9.mat');

cutoffSize = 3; 
useNCL = [1:10];


ixBool2 = ismember(horzcat(exploreAnalysis.nCL),useNCL);


ixBool1 = horzcat(exploreAnalysis.priorAveSize) > cutoffSize;
ea = exploreAnalysis(ixBool1 & ixBool2);


figure;
histogram(horzcat(ea.probMassHInD),100);title('Prob H in D');

figure;
histogram(horzcat(ea.probMassDInH),100);title('Prob D in H');

figure;
histogram(horzcat(ea.probMassPInH),100);title('Prob P in H');

figure;
histogram(horzcat(ea.probMassHInP),100);title('Prob H in P');

figure;hist(horzcat(ea.scoreDMedianDiff));
title('Difference of median scores not in H and D');


figure;hist(horzcat(ea.scorePMedianDiff));
title('Difference of median scores not in H and P');