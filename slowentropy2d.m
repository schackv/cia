function [H, dH,p] = slowentropy2d(x,y, Sigma, Dx,Dy)
% SLOWENTROPY1D One-dimensional entropy by pairwise distances
% A Gaussian kernel with the given bandwidth covariance matrix is used


% warning('Skal checkes for fejl!');

if nargin<5
     D = squareform(pdist([x y],'mahalanobis',Sigma));
%     Dy = squareform(pdist(y));
end

N = length(x);

p = 1/(2*pi*sqrt(det(Sigma))) * exp(-0.5 * D.^2);

H = -1/N*sum(log(1/N*sum(p,2)+eps)); 
dH=NaN;