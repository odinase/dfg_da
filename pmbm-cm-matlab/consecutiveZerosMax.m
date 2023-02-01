function m = consecutiveZerosMax(n)

f = find(diff([false,n==0,false])~=0);
[m,ix] = max(f(2:2:end)-f(1:2:end-1));