function [xa,Ra] = cmAlternative(polar,R)

%Function to carry out Bar-Shaloms debiased measurement conversion
%Written by Edmund Brekke July 2006
%@ polar:       Measurement vector in polar coordinates
%@ R:           Constant measurement noise covariance in polar coordinates
%> xa:          Converted measurement in cartesian coordinates
%> Ra:          Corresponding covariance matrix

rm = polar(1);
thetam = polar(2);

R11a = rm^2*R(2,2)*sin(thetam)^2 + R(1,1)*cos(thetam)^2;
R22a = rm^2*R(2,2)*cos(thetam)^2 + R(1,1)*sin(thetam)^2;
R12a = (R(1,1)-rm^2*R(2,2))*sin(thetam)*cos(thetam);

Ra = [R11a,R12a;R12a,R22a];

xa = [rm*cos(thetam);rm*sin(thetam)];