function xList = cubicSolve(a,b,c)

% Asumme equation written in form x^3 + a*x^2 + b*x + c = 0

p = b - a^2/3;
q = c + (2*a^3-9*a*b)/27;

shiftPlus = -1/2 + i*sqrt(3)/2;
shiftMinus = -1/2 - i*sqrt(3)/2;

yCubicPlus = (-q + sqrt(q^2 + 4*p^3/27))/2;

yPlus1 = yCubicPlus^(1/3);
yPlus2 = yCubicPlus^(1/3)*shiftPlus;
yPlus3 = yCubicPlus^(1/3)*shiftMinus;

yList = [yPlus1,yPlus2,yPlus3];

xList = -p./(3*yList) + yList - a/3;