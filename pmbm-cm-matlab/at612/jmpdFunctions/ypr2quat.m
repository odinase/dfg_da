function q = ypr2quat(eulerArr)

% Converting Euler angles to quaternions
% Written by Edmund Brekke during October 2010
% @eulerArr:    Euler angles in XYZ convention
% >q:           Corresponding quaternions
% $:            quatMultiply

% Euler angles are given according to intrinsic rotation order X <- Y <- Z
% ... which corresponds to extrinsic order R_Z * R_Y * R_X
% ... i.e. yaw first, then pitch around new y-axis, then roll around final x-axis.

% All Euler angles are positive CCW around the corresponding axes.

roll = eulerArr(1,:);
pitch = eulerArr(2,:);
yaw = eulerArr(3,:);

cR = cos(roll/2);
sR = sin(roll/2);
cP = cos(pitch/2);
sP = sin(pitch/2);
cY = cos(yaw/2);
sY = sin(yaw/2);
q = [cR.*cP.*cY + sR.*sP.*sY; ...
    sR.*cP.*cY - cR.*sP.*sY; ...
    cR.*sP.*cY + sR.*cP.*sY; ...
    cR.*cP.*sY - sR.*sP.*cY];
    