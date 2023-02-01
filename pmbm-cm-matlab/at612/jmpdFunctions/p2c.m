function c = p2c(p)

%Function to transform polar coordinates into cartesian coordinates
%Written by Edmund Brekke July 2006
%@ p:       Vectors in polar coordinates
%> c:       Vectors in cartesian coordinates

nt = size(p,2);
c = zeros(2,nt);

r = p(1,:);
theta = p(2,:);

c(1,:) = r.*cos(theta);
c(2,:) = r.*sin(theta);
