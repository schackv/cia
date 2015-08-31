function [p,H, dH] = slowentropy1d(x, sigma, D)
% SLOWENTROPY1D One-dimensional entropy by pairwise distances
% A Gaussian kernel with the given bandwidth is used


% warning('Skal checkes for fejl!');

if nargin<3
    D = squareform(pdist(x));
end

N = length(x);

p = 1/(sqrt(2*pi)*sigma) * exp(- 1/(2*sigma^2) * D.^2 );    % p_ij = p(x_j|x_i)
if nargout>1
    H = -1/N*sum(log(1/N*sum(p,2)+eps));
end
if nargout>2
    dH=NaN;
end