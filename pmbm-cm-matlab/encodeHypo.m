function x = encodeHypo(h,maxM)

% Make each column in h be represented by a single number

nSlots = size(h,2);
x = zeros(1,nSlots);
h(isnan(h)) = maxM+1;

powbase = maxM+2;

for ii=1:nSlots
   for jj=1:size(h,1)
       x(ii) = x(ii) + h(jj,ii)*powbase^(jj-1);
   end
end

