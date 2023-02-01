load('guromat_all.mat');

datamat = a';

nearThreshold = 10;

n = size(datamat,2);

ix = datamat(1,:) > nearThreshold;

datamat2 = datamat(:,ix);
aves = sqrt(mean(datamat2.^2,2));
datamat3 = datamat2./repmat(aves,[1,size(datamat2,2)]);

% P = covarianceEmpirical(datamat3);
% figure;
% imagesc(P);

figure;
plot(datamat2(2,:),datamat2(7,:),'.');