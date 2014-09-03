function [r, phi] = cart2hypersph(X)
% CART2HYPERSPH Convert cartesian coordinates to hyperspherical
% coordinates.
%
% Inputs
%       X       Cartesian coordinates of size N x n
%
% Outputs
%       r       Radius part (N x 1)
%       phi     Angular coordinates (N x (n-1)).
%               The range of phi(:,1:end-1) is [0, 2*pi]
%

% Jacob S. Vestergaard
% www.imm.dtu.dk/~jsve
% Last edit: 22/11/2011

cumr = fliplr(sqrt( cumsum( fliplr(X).^2, 2) ));
r = cumr(:,1);

phi = [acotmod(X(:,1:end-2) ./ cumr(:,2:end-1)) , ...
     2*acotmod( (cumr(:,end-1) + X(:,end-1) ) ./ X(:,end))];

end

function p = acotmod(phi)
% ACOTMOD acot modified
% Returns acot(phi) for phi >= 0 and pi + acot(phi) for phi<0

p = pi*(phi<0) + acot(phi);

end