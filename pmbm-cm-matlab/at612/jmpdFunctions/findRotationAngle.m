function angle = findRotationAngle(c,a,b)

% Written by Edmund Brekke long ago (Spring 2008)
% Revised by Edmund Brekke during June 2011
% @c:       Axis of rotation (only single vector)
% @a:       Vector where rotation begins (only single vector)
% @b:       Vector where rotation ends (may contain multiple vectors)
% >angle:   Rotation angle from a to b around c.

c = c/norm(c);
a = a/norm(a);
b = b./repmat(sqrt(b(1,:).^2+b(2,:).^2+b(3,:).^2),[3,1]);
    


% First construct orthonormal basis for complement of c.

A = [c,a,cross(a,c)];

[Q,temp] = qr(A);
u1 = Q(:,2);
u2 = Q(:,3);

%u1 = (a'*c)*a - cross(c,a);
%u2 = (u1'*c)*u1 - cross(c,u1);
u1 = u1/norm(u1);
u2 = u2/norm(u2);

% Project a and b onto complement of c.

q = [u1,u2];
p = q*q';
aProj = p*a;
bProj = p*b;

% basisTest1 = dot(u1,c)
% basisTest2 = dot(u2,c)
% basisTest3 = dot(u1,u2)

% Determine signs of rotation angles

crossAB = cross(repmat(a,[1,size(b,2)]),b);
crossProj = dot(crossAB,repmat(c,[1,size(crossAB,2)]));
mySign = sign(crossProj);



% Calculate values of rotation angles

angle = acos(dot(repmat(aProj,[1,size(bProj,2)]),bProj)...
    ./(repmat(sqrt(aProj(1,:).^2+aProj(2,:).^2+aProj(3,:).^2),...
[1,size(bProj,2)]).*sqrt(bProj(1,:).^2+bProj(2,:).^2+bProj(3,:).^2)))...
.*mySign;


preAngle = dot(repmat(aProj,[1,size(bProj,2)]),bProj)...
    ./(repmat(sqrt(aProj(1,:).^2+aProj(2,:).^2+aProj(3,:).^2),...
[1,size(bProj,2)]).*sqrt(bProj(1,:).^2+bProj(2,:).^2+bProj(3,:).^2));

%aProj
%bProj(:,60)
% figure;
% for k=1:size(bProj,2)
%     plot(bProj(1,k),bProj(3,k),'.','color',[0,0,k/size(bProj,2)]);
%     hold on;
% end


%figure;plot(preAngle,'b.');title('preAngle');
%
% figure;plot(mySign);
%figure;plot(angle,'r.');
