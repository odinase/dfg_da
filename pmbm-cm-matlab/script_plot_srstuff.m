load tempFileCM.mat

set(0,'defaulttextinterpreter','latex')
%set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'DefaultTextInterpreter', 'latex')
set(groot, 'defaultLegendInterpreter','latex');

pTrueAve = mean(pTrueDmc(:,:,dd),2);pBestAve = mean(successOrFailureDmc(:,:,dd),2);
figure;plot(pTrueAve,'r');hold on;plot(pBestAve,'m-.');hold on;plot(pBestAve.^2,'m:');


pTrueAve = mean(pTrueAveAngel(:,:,dd),2);pBestAve = mean(successOrFailureDmcAngel(:,:,dd),2);
plot(pTrueAve,'b');hold on;plot(pBestAve,'c-.');hold on;plot(pBestAve.^2,'c:');

hold on;



hold off;

legend('SR with CM','AMP with CM','AMP$^2$ with CM','SR without CM','AMP without CM','AMP$^2$ without CM');
xlabel('Time step $k$');
ylabel('Probability');