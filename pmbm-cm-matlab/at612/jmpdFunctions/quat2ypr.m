function [eulerArr] = quat2ypr(q)

% Converting quaternions to Euler angles
% Written by Edmund Brekke during October 2010
% @q:           4 x N array of quaternions
% >eulerArr:    3 x N array of Euler angles

% Euler angles are given according to intrinsic rotation order X <- Y <- Z
% ... which corresponds to extrinsic order R_Z * R_Y * R_X
% ... i.e. yaw first, then pitch around new y-axis, then roll around final x-axis.

% All Euler angles are positive CCW around the corresponding axes.

q0 = q(1,:,:);
q1 = q(2,:,:);
q2 = q(3,:,:);
q3 = q(4,:,:);

roll = atan2(2*(q3.*q2+q0.*q1),q0.^2-q1.^2-q2.^2+q3.^2);
pitch = asin(2*(q0.*q2-q1.*q3));
yaw = atan2(2*(q1.*q2+q0.*q3),q0.^2+q1.^2-q2.^2-q3.^2);

eulerArr = [roll;pitch;yaw];

