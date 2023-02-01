function val = setdensityBernoulliPoisson(x,exi,mu,cv,rate,muPoiss,cvPoiss)


card = size(x,2);

% Generate and evaluate the card+1 Bernoulli-Poisson hypotheses

vPartLogs = normpdfLog(x,muPoiss,cvPoiss)+log(rate);

contributions = zeros(1,card);
for ii=1:card
    gPartLog = log(exi) + normpdfLog(x(:,ii),mu,cv);
    nonI = setdiff(1:card,ii);
    vPL = sum(vPartLogs(nonI));    
    contributions(ii) = exi*exp(sum(vPL)+gPartLog);
end
lastTerm = exp(sum(vPartLogs))*(1-exi);
val = exp(-rate)*(sum(contributions) + lastTerm);



