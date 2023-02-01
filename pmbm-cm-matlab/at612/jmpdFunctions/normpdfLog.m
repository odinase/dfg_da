function value = normpdfLog(x,mu,sigma,sigmaInv)

% Calculates logarithm of normal pdf given covariance and its inverse
% Written by Edmund Brekke during March 2010
% Corrected by Edmund Brekke during January 2011
% Modified by Edmund Brekke during May 2011
% @x:           Stochastic vector
% @mu:          Expectation
% @sigma:       Covariance matrix
% @sigmaInv:    Inverse covariance matrix (optional)
% >value:       Logarithm of normal pdf

% The main purpose of this function is to allow both x and mu to be a
% single column vector or a matrix of column vectors. This necessarily
% requires some boolean logic as used below.

mydet = det(sigma);
p = size(x,1);

if(size(x,2) > size(mu,2) && size(mu,2) == 1)
    xArr = x;
    muArr = repmat(mu,[1,size(x,2)]);
elseif(size(mu,2) > size(x,2) && size(x,2) == 1)
    muArr = x;
    xArr = repmat(x,[1,size(mu,2)]);
else
    muArr = mu;
    xArr = x;
end

logVal = -log((2*pi)^(p/2)*sqrt(mydet));
nuArr = (xArr-muArr);
if(nargin == 3)
   expVal = -sum((nuArr'/sigma)'.*nuArr,1)/2; 
else
   expVal = -sum((nuArr'*sigmaInv)'.*nuArr,1)/2; 
end



value = logVal + expVal;
