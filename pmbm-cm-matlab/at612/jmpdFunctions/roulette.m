function indicesout = roulette(weights,k)

cumweights = cumsum(weights);

m = length(cumweights);

indicesout = zeros(k,1);
noise = rand(k,1);
for(i=1:k)
    test = 1;
    j=1;
    while(test == 1 && j <= m)
        if(noise(i) <= cumweights(j))
            indicesout(i)=j;
            test = 0;
        else
            ;
        end
        j = j+1;
    end
end

