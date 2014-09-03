function [Xs,T] = sphere(X,T)
% SPHERE Sphere data to orthogonal bases with unit variance
% Can also "unsphere" data using a given transformation matrix.
% 
% Inputs
%   X       Data 1
%   Y       Data 2
%   T       (optional) Transformation matrix


[N,p] = size(X); 

X = zscore(X);      % Center and standardize
if nargin==1    % Sphere
    S = cov(X);
    T = inv(chol(S));
    Xs = X*T;
else
    Xs = X*T;   % "unsphere"
end
