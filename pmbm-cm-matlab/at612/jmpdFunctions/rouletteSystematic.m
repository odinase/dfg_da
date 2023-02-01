function indicesout = rouletteSystematic(weights,k)

cumweights = cumsum(weights);
m = length(cumweights);

indicesout = zeros(k,1);
noise = rand(1,1)/k;

i = 1;
for j=1:k
    uj = noise + (j-1)/k;    
    while uj > cumweights(i)
        i = i + 1;      
    end
    indicesout(j) = i; 
end
temp = randperm(size(indicesout,1));
indicesout = indicesout(temp);