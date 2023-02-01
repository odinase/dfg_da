function y = bump_dundas(t,center,width)

% Smooth bump function based on Dundas chapter 7
% Written by Edmund Brekke 20th of November 2006
% @t:       input vector
% @center:  center point of the bump
% @width:   effective width of the bump
% @y:       bump function


windowmin = center - width;
windowmax = center + width;

qix = find(t < windowmax & t > windowmin);
q = (t(qix)-windowmin)/(2*width);

y1 = zeros(min(qix)-1,1)';
y2 = bump_alpha(1,q);
y3 = ones(length(t)-max(qix),1)';
y = [y1,y2,y3];

function alpha = bump_alpha(epsilon,t)

beta = bump_lambda(t).*bump_lambda(epsilon-t);
beta = beta/sum(beta);
alpha = cumsum(beta)/sum(beta);

function lambda = bump_lambda(t)

% Based on Dundas chapter 7.2 page 94

index_t_not_zero = t ~= 0;

lambda = zeros(size(t));
lambda(index_t_not_zero) = max(exp(-1./t(index_t_not_zero).^2),0);
lambda(t <= 0) = 0;