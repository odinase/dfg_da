function hessMatrix = hessFD(fun,x,dx)

hessMatrix = zeros(length(x));

for ii=1:length(x)
    for jj=1:length(x)
        
        
        firstVec = zeros(length(x),1);
        secondVec = zeros(length(x),1);
        thirdVec = zeros(length(x),1);
        firstVec(ii) = firstVec(ii) + dx;
        firstVec(jj) = firstVec(jj) + dx;
        secondVec(ii) = secondVec(ii) + dx;
        thirdVec(jj) = thirdVec(jj) + dx;

        
        hessMatrix(ii,jj) = fun(x+firstVec) - fun(x+secondVec) - fun(x+thirdVec) + fun(x);
        
        
    end
end
hessMatrix = hessMatrix/dx^2;
