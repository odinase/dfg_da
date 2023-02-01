function [dfg] = jacobianP2C(xS)


r = xS(1);
theta = xS(2);

dfg = [cos(theta),-r*sin(theta); sin(theta), r*cos(theta)];




%x = xB(1);
%y = xB(2);
%rho = norm(xB);
%dfg = [x/rho,y/rho; -y/rho^2, x/rho^2];

