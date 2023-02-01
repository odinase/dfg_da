set(0,'defaulttextinterpreter','latex')
%set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'DefaultTextInterpreter', 'latex')
set(groot, 'defaultLegendInterpreter','latex');

figure;plot(sqrt(mean(ospaDmc(:,:,5),2)),'k');
hold on;
plot(sqrt(mean(ospaDmcAngel(:,:,5),2)),'k-.');
hold off;
xlabel('Time step $k$');
ylabel('RMS GOSPA');
legend('PMBM with CM','PMBM without CM');


figure;plot(sqrt(mean(ospaMisDmc(:,:,5),2)),'k');
hold on;
plot(sqrt(mean(ospaMisDmcAngel(:,:,5),2)),'k-.');
hold off;
xlabel('Time step $k$');
ylabel('RMS Mis-GOSPA');
legend('PMBM with CM','PMBM without CM');

figure;plot(sqrt(mean(ospaFalseDmc(:,:,5),2)),'k');
hold on;
plot(sqrt(mean(ospaFalseDmcAngel(:,:,5),2)),'k-.');
hold off;
xlabel('Time step $k$');
ylabel('RMS False Track GOSPA');
legend('PMBM with CM','PMBM without CM');


figure;plot(sqrt(mean(ospaLocDmc(:,:,5),2)),'k');
hold on;
plot(sqrt(mean(ospaLocDmcAngel(:,:,5),2)),'k-.');
hold off;
xlabel('Time step $k$');
ylabel('RMS Localization GOSPA');
legend('PMBM with CM','PMBM without CM');