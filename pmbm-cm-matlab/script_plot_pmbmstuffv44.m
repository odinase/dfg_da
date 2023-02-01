set(0,'defaulttextinterpreter','latex')
%set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'DefaultTextInterpreter', 'latex')
set(groot, 'defaultLegendInterpreter','latex');

x=vec(squeeze(timesArrExt(6,:,:)));
[~,edges] = histcounts(log10(x));histogram(x,10.^edges); set(gca, 'xscale','log')
set(gca, 'yscale','log')
axis([0.001,700,0.6,3000])
title('Histogram of runtime for Double Murty');
xlabel('Time [s]');
ylabel('Histogram count (15000 samples in total)');


figure;
x=vec(squeeze(timesArrExt(8,:,:)));
[~,edges] = histcounts(log10(x));histogram(x,10.^edges); 
set(gca, 'xscale','log');set(gca, 'yscale','log');
axis([0.001,700,0.6,3000])
title('Histogram of runtime for cluster splitting');
xlabel('Time [s]');
ylabel('Histogram count (15000 samples in total)');


fig1 = figure;
plot(1:nK,pBestAve,'b-o');
hold on;
plot(1:nK,pTrueAve,'k-*');
plot(1:nK,pBestAve.^2,'r-d');
hold off;
legend({'$S$','$A$','$S^2$'}, 'interpreter', 'latex','fontsize',12);
legend boxoff 
xlabel('Time step $k$');
ylabel('Probabilities');
set(fig1, 'PaperPositionMode', 'auto');
