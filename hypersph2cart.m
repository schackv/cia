function X = hypersph2cart(r,phi)
% HYPERSPH2CART Converts spherical coordinates to cartesian.
%
% Inputs 
%       r       Radius, vector of length N or scalar. If scalar, r =
%               ones(N,1) * r.
%       phi     Angular coordinates of size N x (p-1).
%               phi(:,end) ranges over [0,2*pi] while phi(:,1:end-1) ranges
%               over [0,pi]
%
% Outputs
%       X       Matrix of cartesian coordinates of size N x p
%

% Jacob S. Vestergaard
% www.imm.dtu.dk/~jsve
% Last edit: 22/11/2011

[N,p] = size(phi);

if isscalar(r)
    r = ones(N,1)*r;
else
    r = r(:);
end

phi_cos = [cos(phi), ones(N,1)];
phi_sin = [ones(N,1), cumprod(sin(phi),2)];

X = bsxfun(@times,r, phi_sin .* phi_cos);
