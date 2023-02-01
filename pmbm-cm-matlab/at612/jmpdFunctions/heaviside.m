function Y = heaviside(X)
% Heaviside(X) = 0, for X < 0
% = 1, for X > 0
Y = zeros(size(X));
Y(X > 0) = 1;
Y(X == 0) = 1/2;