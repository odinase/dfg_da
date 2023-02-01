function [wreduced, xreduced, Preduced, costs, merges] = Runnalls(w, x, P, m, maxD, noSmaller)
% RUNNALLS Reduce a Gaussian mixture using Runnalls algorithm
%   [wred, xred, Pred, costs, merges] = Runnalls(w, x, P, m)
%   reduces a Gaussian mixture described by the weights w, means x and
%   covariances P of sizes (n x 1), (n, dim) and (n x dim x dim),
%   respectively, to a mixture with min(n, m) components by merging similar
%   components through moment matching.
%
%   [wred, xred, Pred, costs, merges] = Runnalls(..., maxD)
%   stops reducing the mixture if the upper bound on
%   the next merges divergence will be more than maxD, even thoug the
%   mixture has more than m components. MaxD can be set to inf if one does
%   not want an upper Bound.
%
%   [wred, xred, Pred, costs, merges] = Runnalls(..., maxD, noSmaller)
%   continues to merge components even if there are less than m. Stops when
%   the upper bounded divergence will be more than noSmaller on the next
%   merge. In the strange event that maxD < noSmaller, maxD will not be 
%   considered.

% Author: Lars-Christian Ness Tokle
% email: lars-christian.n.tokle@ntnu.no
% Website: https://www.ntnu.no/ansatte/lars-christian.n.tokle
% May 2020; Last revision: 11-May-2020

if nargin < 5
    maxD = inf;
end
if nargin < 6
    noSmaller = 0;
end
N = size(w, 1);
logDetPs = zeros(N,1);

% precompute logdets
for i = 1:N
    logDetPs(i) = logdetPSD(squeeze(P(i, :, :)));
end

dists = zeros(N-1, N);
minD = inf;
minI = -1;
minJ = -1;

% find all pairwise Runnalls dists, and store smallest
for i = 1:N-1
    for j = (i+1):N
        dists(i, j) = Bprelog(...
            w(i), x(i, :), squeeze(P(i, :, :)), logDetPs(i),...
            w(j), x(j, :), squeeze(P(j, :, :)), logDetPs(j));
        if dists(i, j) < minD
            minD = dists(i, j);
            minI = i;
            minJ = j;
        end
    end
end
% merge 2 and 2 components until requirements are satisfied
idxs = 1:N;
costs = [];
merges = [];
while (length(idxs) > m && minD < maxD) || minD < noSmaller
    costs = [costs; minD];
    merges = [merges; minI, minJ];
    % new component
    w1 = w(minI);
    w2 = w(minJ);
    x1 = x(minI, :);
    x2 = x(minJ, :);
    P1 = squeeze(P(minI, :, :));
    P2 = squeeze(P(minJ, :, :));

    wnew = w1 + w2;
    xd  = (x1 - x2)/wnew;
    munew = (w1 * x1 + w2 * x2)/wnew;
    Pnew = (w1 * P1 + w2 * P2)/wnew...
        + w1*w2 * (xd' * xd); % xd is a row vector, hence xd' * xd

    % store new component in the lowest idx, ie. minI < minJ
    w(minI) = wnew;
    x(minI, :) = munew;
    P(minI, :, :) = Pnew;
    logDetPs(minI) = logdetPSD(Pnew);
    % remove old component from being considered
    dists(1:minJ, minJ) = inf;
    idxs = idxs(idxs ~= minJ);
    
    % calculate dists to new component from idx smaller than minI
    for k = 1:N
        %, i in enumerate(idxs[:-1]):
        i = idxs(k);
        if i >= minI
            if i == minI
                k = k + 1;
            end
            break
        end
        dists(i, minI) = Bprelog(...
            w(i), x(i, :), squeeze(P(i, :, :)), logDetPs(i),...
            w(minI), x(minI, :), squeeze(P(minI, :, :)), logDetPs(minI));
    end

    % calculate dists to new component from idx greater than minI
    for j = idxs(k:end)
        dists(minI, j) = Bprelog(...
            w(minI), x(minI, :), squeeze(P(minI, :, :)), logDetPs(minI),...
            w(j), x(j, :), squeeze(P(j, :, :)), logDetPs(j));
    end

    % find new smallest to merge
    minD = inf;
    for i = idxs(1:(end - 1))
        [disti, minIidx] = min(dists(i, (i + 1):end));
        minIidx = minIidx + i; % compensate for using triu mat.
        if disti < minD
            minD = disti;
            minI = i;
            minJ = minIidx;
        end
    end
end
wreduced = w(idxs);
xreduced = x(idxs, :);
Preduced = P(idxs, :, :);
end


function b = B(w1, mu1, P1, w2, mu2, P2)
    w12 = w1 + w2;
    w1_12 = w1/w12;
    w2_12 = w2/w12;
    
    mud = mu1 - mu2;

    P12 = (w1_12 * P1 + w2_12 * P2 + w1_12 * w2_12 * (mud' * mud));

    logdet1 = logdetPSD(P1);
    logdet2 = logdetPSD(P2);
    logdet12 = logdetPSD(P12);

    b = 0.5 * (w12 * logdet12 - w1 * logdet1 - w2 * logdet2);
end

function b = Bprelog(w1, mu1, P1, logdet1, w2, mu2, P2, logdet2)
    w12 = w1 + w2;
    w1_12 = w1/w12;
    w2_12 = w2/w12;

    mudiff = (mu1 - mu2);
    P12 = (w1_12 * P1 + w2_12 * P2...
        + w1_12 * w2_12 * (mudiff' * mudiff)); % xd is a row vector, hence xd' * xd

    logdet12 = logdetPSD(P12);

    b = 0.5 * (w12 * logdet12 - w1 * logdet1 - w2 * logdet2);
end

function logdet = logdetPSD(A)
    logdet = 2 * sum(log(diag(chol(A))));
end

function [xbar, Pbar] = gaussianMixtureMoments(w, x, P)
xbar = sum(w * x, 1);
Pint = sum(w * P, 1);

xinnov = x - xbar;

Pext = zeros(size(Pint));
for i = 1:length(w)
    Pext = Pext + w(i) * (xinnov(i, :)' * xinnov(i, :)); % xd is a row vector, hence xd' * xd
end

Pbar = Pint + Pext;
end