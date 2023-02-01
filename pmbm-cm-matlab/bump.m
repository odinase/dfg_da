function y = bump(x,lower,upper,pre,post,direction)

s = (x-pre)/(post-pre);
if(direction == -1)
    t = 1 - s;
else
    t = s;
end

lowerLogicals = t <= 0;
middleLogicals = t > 0 & t < 1;
upperLogicals = t >= 1;

f = zeros(size(x));
f(lowerLogicals) = 0;
f(middleLogicals) = 1./(1+exp(1./t(middleLogicals))./exp(1./(1-t(middleLogicals))));
f(upperLogicals) = 1;

y = f*(upper-lower) + lower;