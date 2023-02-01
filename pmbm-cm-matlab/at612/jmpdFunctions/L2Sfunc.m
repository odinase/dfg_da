function polar = L2Sfunc(cartesian,x0,theta0)

%Function to transform from standard cartesian space to restricted polar space
%Written by Edmund Brekke may 2006
%@ cartesian:   vector [x;y] in standard cartesian space
%@ x0:          the point in cartesian space around which polar space is defined
%@ theta0:      the standardised angle corresponding to theta=0 in polar space
%@ sensorangle: the width of polar space
%§ p2c.m
%> polar:       transformed vector [r;theta] in polar space defined by x0 and theta0

m = size(cartesian,2);

polar = c2p(cartesian-x0*ones(1,m));
polar(2,:) = mod(polar(2,:) - theta0*ones(1,m),2*pi);

%sensormaxtheta = mod(sensorangle + theta0,2*pi);

%If cartesian is not in visible area multiply with NaN

% for(j=1:m)
%     if(mod(polar(2,j),2*pi) > mod(sensorangle,2*pi))
% 
%         polar(:,j) = polar(:,j)*NaN;
%     
%     end
% end