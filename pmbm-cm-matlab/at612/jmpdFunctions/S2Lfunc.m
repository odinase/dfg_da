function cartesian = S2Lfunc(polar,x0,theta0)

%Function to transform from polar continuous space to standard cartesian continuous space
%Written by Edmund Brekke may 2006
%@ polar:       vector [r;theta] in polar space given by x0 and theta0
%@ x0:          the point in cartesian space around which polar space is defined
%@ theta0:      the standardised angle corresponding to theta=0 in polar space
%§ p2c.m
%> cartesian:   vector [x;y] in standard cartesian space

m = size(polar,2);
polar(2,:) = mod(polar(2,:) + theta0*ones(1,m),2*pi);
cartesian = p2c(polar) + x0*ones(1,m);