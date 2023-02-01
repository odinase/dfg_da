function mu = nanAverage(xArray,dim)

% Function to evaluate arbitrary means of array containing NaNs
% Written by Edmund Brekke during July / December 2007

% Safe case when MD array cannot be confused with 2D matrix

if((length(size(xArray)) <= 2 && dim < 3) || length(size(xArray)) > 2)

    ix = find(isnan(xArray));

    sizeOfArray = size(xArray);

    xArrayCopy = xArray;
    xArrayCopy(ix) = 0;

    xArrayOnes = ones(sizeOfArray);
    xArrayOnes(ix) = 0;

    mu = sum(xArrayCopy,dim)./sum(xArrayOnes,dim);

% Need special treatment when there is room for confusion

else
    
    mu = xArray;
    
end