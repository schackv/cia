function W = mimatrix(X,varargin) 
% MIMATRIX Mutual information between the columns of X, such that the ij'th
% element of W is the mutual information between X(:,i) and X(:,j).

[m,n] = size(X);
W = NaN(n);
for i=1:n
    for j=i:n
        [I,~,Inorm] = kdeMI(X(:,i),X(:,j),varargin{:});
        W(i,j) = I;
        W(j,i) = W(i,j);
    end
end

% Normalize to identity diagonal
% identitymi = sqrt((diag(W)*ones(1,n)) .* (ones(n,1)*diag(W)'));
% W = W ./identitymi;
