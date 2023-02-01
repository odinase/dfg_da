function q = quatAdjoint(p)

q = zeros(4,size(p,2));
q(1,:) = p(1,:);
q(2:4,:) = - p(2:4,:);
