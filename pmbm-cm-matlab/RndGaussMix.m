%% params
dim = 2; % dimension of problem
lambda = 8; % Poisson: expected number of components
weight_variability = 2; % >0, this is 1/(1 - dirichlet_param)
Spread = 10 * eye(dim); % spread of the means of the components
Sigma = eye(dim); % expected covariance of components
df = dim + 1; % degrees of freedom for sampling covariance from Wishart
m = 4; % number of components to reduce this to using Runnals

%% sample Gaussian mixture
n = poissrnd(lambda);
w = dirichletRnd(ones(n, 1)/weight_variability + 1);
mu = (chol(Spread) * randn(dim, n))';
P = zeros(n, dim, dim);
for i = 1:n
    P(i, :, :) = wishrnd(Sigma/df, df);
end

%% reduce the mixture
[wred, mured, Pred, costs, merges] = Runnalls(w, mu, P, m);
assert(abs(sum(wred) - 1) < 100*eps);
costs
merges
%% create plot grid
[xbar, Pbar] = gaussianMixtureMoments(w, mu, P);
[xbarred, Pbarred] = gaussianMixtureMoments(wred, mured, Pred);
assert(sum(abs(xbar - xbarred)) < 100 * eps); % test momentpreserving
assert(sum(sum(abs(Pbar - Pbarred))) < 100 * eps); % test momentpreserving
xspace = linspace(-1 * Pbar(1,1), 1 * Pbar(1,1), 100) + xbar(1);
yspace = linspace(-1 * Pbar(2,2), 1 * Pbar(2,2), 100) + xbar(2);
[xx, yy] = meshgrid(xspace, yspace);
XX = [xx(:), yy(:)];

%% evaluate original pdf
VALS = evalGaussMix(XX, w, mu, P);
vals = reshape(VALS, size(xx));

%% evaluate reduced pdf
VALSred = evalGaussMix(XX, wred, mured, Pred);
valsred = reshape(VALSred, size(xx));

%% plot the pdfs and difference of logs
figure(1); clf;
subplot(2, 2, 1);
surfc(xx, yy, vals, 'FaceColor', 'interp');
title(sprintf('n = %d', n))
subplot(2, 2, 2);
surfc(xx, yy, valsred, 'FaceColor', 'interp');
title(sprintf('Runnalls, m = %d', m))
subplot(2, 2, 3);
surfc(xx, yy, log(vals./valsred), 'FaceColor', 'interp')
zlabel('log(p/q)');

%% plot ellipses
subplot(2, 2, 4);
hold on; grid on;
title('ellipses (red might hide blue if not altered');
scatter(mu(:, 1), mu(:, 2), 'b');
scatter(mured(:, 1), mured(:, 2), 'r');
th = (0:100)*2*pi/100;
circle = [cos(th); sin(th)];
for i = 1:n
    ell = mu(i, :)' + chol(squeeze(P(i, :, :)))' * circle;
    text(mu(i, 1), mu(i, 2), sprintf('%d', i));
    plot(ell(1, :), ell(2, :), 'b');
end
for i = 1:length(wred)
    ell = mured(i, :)' + chol(squeeze(Pred(i, :, :)))' * circle;
    plot(ell(1, :), ell(2, :), 'r');
end

%% functions needed
function val = evalGaussMix(x, w, mu, P)
n = length(w);
val = zeros(size(x, 1), 1);
for i = 1:n
    for j = 1:size(x, 1)
        Pi = squeeze(P(i, :, :));
        NIS = (x(j, :) - mu(i,:)) * (Pi\(x(j, :) - mu(i,:))');
        val(j) = val(j) + w(i)*exp(-0.5*NIS)/sqrt(det(2 * pi * Pi));
    end
end
end

function x = dirichletRnd(a, m)
% Generate samples from a Dirichlet distribution.
% Input:
%   a: k dimensional vector
%   m: k dimensional mean vector
% Output:
%   x: generated sample x~Dir(a,m)
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 2
    a = a*m;
end
x = gamrnd(a,1);
x = x/sum(x);
end

function [xbar, Pbar] = gaussianMixtureMoments(w, x, P)
xbar = sum(w .* x, 1);
Pint = squeeze(sum(w .* P, 1));

xinnov = x - xbar;

Pext = zeros(size(Pint));
for i = 1:length(w)
    Pext = Pext + w(i) * (xinnov(i, :)' * xinnov(i, :));
end

Pbar = Pint + Pext;
end