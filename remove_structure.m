function Xhat = remove_structure(X,v)
% REMOVE_STRUCTURE Remove structure by inserting white noise (mean 0,
% variance 1) along the direction v.

N = size(X,1);
U = vec2orth(v);
Z = X*U;
Z(:,1) = randn(N,1);
Xhat = Z*U';    % Rotate corrupted/structure-removed data back
