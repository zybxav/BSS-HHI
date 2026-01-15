function y = randsample(x, k, replacement)
%RANDSAMPLE  Simple replacement for MATLAB Statistics Toolbox randsample
%   y = randsample(x, k)
%       randomly sample k values from vector x (without replacement)
%
%   y = randsample(x, k, true)
%       sample with replacement
%
%   x must be a vector

    % argument check
    if nargin < 3
        replacement = false;
    end

    x = x(:);  % ensure column vector
    n = numel(x);

    if ~replacement
        % without replacement
        if k > n
            error('Sample size k (%d) must be <= length(x) (%d).', k, n);
        end
        idx = randperm(n, k);
        y = x(idx);
    else
        % with replacement
        idx = randi(n, k, 1);
        y = x(idx);
    end
end
