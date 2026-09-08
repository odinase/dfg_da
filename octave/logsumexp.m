function s = logsumexp(a, dim)
% Shim for the lightspeed toolbox function of the same name, which
% compute_results.m depends on but is not vendored in this repo.
% s = log(sum(exp(a), dim)), computed with the max shifted out.
  if nargin < 2
    dim = 1;
  end
  mx = max(a, [], dim);
  mx(~isfinite(mx)) = 0;
  s = mx + log(sum(exp(bsxfun(@minus, a, mx)), dim));
end
