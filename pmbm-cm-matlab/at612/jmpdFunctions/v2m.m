function candmat = v2m(candvec,xmax)

%Given vector representation of image pixels find matrix representation
%x is first dimension, i.e. downwards
% Corrected by Edmund Brekke during January 2008

candmat = zeros(2,size(candvec,2));
candmat(1,:) = mod(candvec-1,xmax)+1;
candmat(2,:) = ceil(candvec/xmax); 