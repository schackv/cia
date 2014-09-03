function [H,dH,p] = fastentropy1d(S,sigma,nbins)
% FASTENTROPY1d Calculate marginal entropy estimate of univariate signal
% using a Gaussian kernel density estimator.
%
% Inputs
%       S       N-vector of observations
%       sigma   Standard deviation of Gaussian kernel
%       nbins   (optional) The number of bins used for uniform sampling.
%               Default = 256.
% Outputs
%       H       Entropy estimate
%

% Reference: 
% Shwartz et al, Fast kernel entropy estimation and optimization, 
% Signal Processing, Volume 85, Issue 5, May 2005, Pages 1045-1058, 
% http://www.sciencedirect.com/science/article/pii/S0165168405000216

% Jacob S. Vestergaard
% http://www.imm.dtu.dk/~jsve
% 06-01-2012

if nargin<3
    nbins = 256;
end

[N,d] = size(S);

if d>1
    error('CIA:OneDonly', 'Only one dimensional variables allowed');
end

S = S(:);
smin = min(S);
smax = max(S);
dbins = (smax-smin)/(nbins-2);
Sc = S - smin;  % Move data to min=0

% Check for sigma to dbins relationship
if (dbins/2)>=8*sigma
    nbins = ceil((smax-smin)/(2*sigma));
    dbins = (smax-smin)/nbins;
    warning('CIA:fastentropy1d','Increasing bin count due to spiky distribution.');
end
    

% Perform voting to get uniform sampled histogram-like PDF
pdf = zeros(nbins,1);   % PDF at m_hash
mhash = floor(Sc./dbins)+1;   % Calculate mhash for all points
assert(sum(mhash==0)==0,'Bin with value 0 should be empty');
assert(sum(mhash==nbins)==0,'Bin with value nbins should be empty');
% mhash(mhash<1) = 1;         % Ensure it is between 1 and nbins-1
% mhash(mhash>nbins-1) = nbins-1;
eta = Sc./dbins - mhash + 1;    % Distance to index
h = 1 - eta;                % Linear interpolation function

for n=1:N
    pdf(mhash(n)) = pdf(mhash(n)) + h(n);
    pdf(mhash(n)+1) = pdf(mhash(n) + 1) + 1 - h(n);
end

% Setup gaussian
tmax = 8*sigma;             % Max width of Gaussian
t = dbins/2:dbins:tmax;     % Distance to center
Lt = length(t);
t = [-fliplr(t) t(2:end)];  % Mirror Gaussian
G = exp(-t.^2/2/sigma.^2);  % Gaussian
G = G/(sum(G)*dbins*N);     % Normalize
dG = -(1/sigma^2).*t.*G;    % First-order Gaussian derivative

% Convolve PDF/histogram/uniform sampling with Gaussian
temp = conv(pdf,G);
temp = temp(:);
pdf_g = temp(Lt + (0:nbins-1));   % Extract non-padded part

% Interpolate to original grid and estimate entropy
p = h .* pdf_g(mhash) + (1-h) .* pdf_g(mhash + 1);
H = -1/N * sum(log(p));

if nargout>1
    % Entropy  gradient
    temp    = conv(pdf,dG);
    temp    = temp(:);
    pdf_dg  = temp(Lt + (0:nbins-1));

    dpsi = -1./p./N;
    dpsi_binned = zeros(nbins,1);
    for n=1:N
        dpsi_binned(mhash(n))   = dpsi_binned(mhash(n))   + h(n) * dpsi(n);
        dpsi_binned(mhash(n)+1) = dpsi_binned(mhash(n)+1) + (1 - h(n)) * dpsi(n);
    end
    dp = h .* pdf_dg(mhash) + (1-h) .* pdf_dg(mhash + 1);

    temp = conv(dpsi_binned,dG);    % Convolve with Gaussian derivative
    temp = temp(:);
    dH_binned = temp(Lt + (0:nbins-1));
    dH = dp .* dpsi + h .* dH_binned(mhash) + (1-h) .* dH_binned(mhash + 1);
end
