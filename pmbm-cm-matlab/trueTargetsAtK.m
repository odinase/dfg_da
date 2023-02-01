function [states,labels,measurements] = trueTargetsAtK(targetsTrue,hTrue,k)


if(~isempty(targetsTrue))
    dimX = size(targetsTrue(1).x,1);
    nTarEntire = size(hTrue,2);
    states = NaN*zeros(dimX,nTarEntire);
    labels = NaN*zeros(1,nTarEntire);
    measurements = NaN*zeros(1,nTarEntire);

    for t=1:nTarEntire
        if(ismember(k,targetsTrue(t).k))
            states(:,t) = targetsTrue(t).x(:,targetsTrue(t).k==k);
            labels(:,t) = targetsTrue(t).id;
            measurements(t) = hTrue(k,t);
        end
    end
    ix = ~isnan(states(1,:));
    states = states(:,ix);
    labels = labels(:,ix);
    measurements = measurements(:,ix);
    
    
else
    states = zeros(1,0);
    labels = zeros(1,0);
    measurements = zeros(1,0);    
    
end