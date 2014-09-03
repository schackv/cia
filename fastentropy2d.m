function [H,dH,p] = fastentropy2d(S,sigma,nbins)
% FASTENTROPY2d Calculate joint entropy for a two dimensional sample using
% a Gaussian kernel density estimator. Needs a given estimate of the 
% kernel bandwidth and optionally the number of bins used for 
% uniform sample. 
%
% For marginal entropy use fastentropy1d.
%
% Inputs
%   S       Sample of size N x 2
%   sigma   2-vector of bandwidths of kernel, i.e., standard deviation of 
%           the Gaussian in each direction. Use scalar for isotropic
%           Gaussian.
%   nbins   (optional) Number of bins used for uniform sampling. 
%
% Outputs
%   H       Joint entropy estimate.
%

% This is a generalization to joint entropy from the marginal entropy
% calculation suggested by Shwartz et al in:
% Fast kernel entropy estimation and optimization, Signal Processing, 
% Volume 85, Issue 5, May 2005, Pages 1045-1058, 
% http://www.sciencedirect.com/science/article/pii/S0165168405000216

% Jacob S. Vestergaard
% http://www.imm.dtu.dk/~jsve
% 06-01-2012

if nargin<3
    nbins = 256;
end

if length(sigma)==1
    sigma=[sigma sigma]';
else
    sigma=sigma(:);
end

[N,d] = size(S);

if d~=2, error('CIA:TwoDonly','Only 2-dimensional variables allowed'); end

smin = min(S);
smax = max(S);
dbins = (smax-smin)./(nbins-2);

% Probably more meaningful binning
% % Bin edges
% edges_x = smin(1):dbins(1):smax(1);     edges_x(end) = Inf;
% edges_y = smin(2):dbins(2):smax(2);     edges_y(end) = Inf;
% % Bin centers
% centers_x = edges_x(1:end-1) + dbins(1)*0.5;
% centers_y = edges_y(1:end-1) + dbins(2)*0.5;
% % Do binning
% mhash = zeros(N,2);
% [~,mhash(:,1)] = histc(S(:,1),edges_x);
% [~,mhash(:,2)] = histc(S(:,2),edges_y);
% % Distance to bin
% eta = (S - [centers_x(mhash(:,1))' centers_y(mhash(:,2))'])./repmat(dbins,N,1);
% pdf = zeros(nbins+1,nbins+1);
% 
% 
Sc = S - repmat(smin,N,1);     % Move data to min=(0,0)

% Perform voting to get uniform sampled histogram-like PDF
pdf = zeros(nbins,nbins);
mhash = floor( Sc ./ repmat(dbins,N,1) ) + 1;   % Calculate mhash for all points
assert(sum(mhash(:)<1)==0,'Binning failed, mhash should be nonzero');
assert(sum(mhash(:)>=nbins)==0, 'Binning failed, mhash should be nbins-1 maximum');
% mhash(mhash<1) = 1;                         % Ensure it is between 1 and
% mhash(mhash>nbins-1) = nbins-1;             % nbins-1
eta = Sc ./ repmat(dbins,N,1) - mhash + 1;      % Distance to index in each dimension

% Use bilinear interpolation to create histogram
for n=1:N 
    pdf(mhash(n,1)  ,mhash(n,2))    = pdf(mhash(n,1),  mhash(n,2))   + (1-eta(n,1))*(1-eta(n,2));
    pdf(mhash(n,1)+1,mhash(n,2))    = pdf(mhash(n,1)+1,mhash(n,2))   + eta(n,1)*(1-eta(n,2));
    pdf(mhash(n,1)  ,mhash(n,2)+1)  = pdf(mhash(n,1),  mhash(n,2)+1) + (1-eta(n,1))*eta(n,2);
    pdf(mhash(n,1)+1,mhash(n,2)+1)  = pdf(mhash(n,1)+1,mhash(n,2)+1) + eta(n,1)*eta(n,2);
end

% Setup 2d Gaussian
tmax = 8*sigma;                 % Max width of Gaussian in 2d
t1 = dbins(1)/2:dbins(1):tmax(1);
t1 = [-fliplr(t1) t1(2:end)];
t2 = dbins(2)/2:dbins(2):tmax(2);
t2 = [-fliplr(t2) t2(2:end)];
[T1,T2] = ndgrid(t1,t2);
t = sum( ([T1(:) T2(:)] * diag(1./sigma.^2)) .*[T1(:) T2(:)],2);
T = reshape(t,size(T1));
G = exp(-0.5 *T);           % Equal to mvnpdf([T1(:) T2(:)],[0,0]',SIGMA)* ...
                            %   (2*pi*sqrt(det(SIGMA))), where 
                            %   SIGMA=diag(sigma.^2)
G   = G./(sum(G(:))*prod(dbins)*N);   % Normalize
% Convolve with 2D Gaussian
pdf_g   = imfilter(pdf,G,'same');

% Interpolate to original grid and estimate entropy
% p = zeros(N,1);
% for n=1:N
%     p(n) = (1-eta(n,1))*  (1-eta(n,2)) * pdf_g(mhash(n,1),  mhash(n,2)) + ...
%             eta(n,1)   *  (1-eta(n,2)) * pdf_g(mhash(n,1)+1,mhash(n,2)) + ...
%            (1-eta(n,1))*   eta(n,2)    * pdf_g(mhash(n,1),  mhash(n,2)+1) + ...
%             eta(n,1)   *   eta(n,2)    * pdf_g(mhash(n,1)+1,mhash(n,2)+1);
% end
% Same calculation without for-loop (approx. same time as above)
linind = sub2ind([nbins,nbins],  mhash(:,1)   ,mhash(:,2));
linind2 = sub2ind([nbins,nbins], mhash(:,1)+1 ,mhash(:,2));
linind3 = sub2ind([nbins,nbins], mhash(:,1)   ,mhash(:,2)+1);
linind4 = sub2ind([nbins,nbins], mhash(:,1)+1 ,mhash(:,2)+1);
p =    (1-eta(:,1)).*(1-eta(:,2)) .* pdf_g(linind) ...
          +  eta(:,1).*(1-eta(:,2))    .* pdf_g(linind2) ...
          + (1-eta(:,1)).*eta(:,2)     .* pdf_g(linind3) ...
          + eta(:,1).*eta(:,2)         .* pdf_g(linind4);

H = - 1/N * sum(log(p+eps));

if nargout>1
    % Joint entropy gradient 
    dG  = -bsxfun(@times,diag(1./sigma.^2)*[T1(:) T2(:)]', G(:)');
    dG  = reshape(dG',[size(T1), 2]);     % dx, dy
%     pdf_dg  = cat(3,imfilter(pdf,dG(:,:,1),'same'), ...
%                     imfilter(pdf,dG(:,:,2),'same'));
    pdf_dgx = imfilter(pdf,dG(:,:,1),'same');
    pdf_dgy = imfilter(pdf,dG(:,:,2),'same');
    
    dpsi_binned = zeros(nbins);
%     dp = zeros(N,2);
    dpsi = -1./p./N;
    for n=1:N   
        dpsi_binned(mhash(n,1),mhash(n,2)) = ...
        dpsi_binned(mhash(n,1),mhash(n,2)) + (1-eta(n,1))*  (1-eta(n,2))*dpsi(n);
        dpsi_binned(mhash(n,1)+1,mhash(n,2)) = ...
        dpsi_binned(mhash(n,1)+1,mhash(n,2)) + eta(n,1)   *  (1-eta(n,2))*dpsi(n);
        dpsi_binned(mhash(n,1),mhash(n,2)+1) = ...
        dpsi_binned(mhash(n,1),mhash(n,2)+1) + (1-eta(n,1))*   eta(n,2)*dpsi(n);
        dpsi_binned(mhash(n,1)+1,mhash(n,2)+1) = ...
        dpsi_binned(mhash(n,1)+1,mhash(n,2)+1) +  eta(n,1)   *   eta(n,2)*dpsi(n);
%         dp(n,:) = (1-eta(n,1))*  (1-eta(n,2)) * pdf_dg(mhash(n,1),  mhash(n,2),:) + ...
%                  eta(n,1)   *  (1-eta(n,2)) * pdf_dg(mhash(n,1)+1,mhash(n,2),:) + ...
%                 (1-eta(n,1))*   eta(n,2)    * pdf_dg(mhash(n,1),  mhash(n,2)+1,:) + ...
%                  eta(n,1)   *   eta(n,2)    * pdf_dg(mhash(n,1)+1,mhash(n,2)+1,:);
    end
    dpx = (1-eta(:,1)).*(1-eta(:,2)) .* pdf_dgx(linind) ...
          +  eta(:,1).*(1-eta(:,2))    .* pdf_dgx(linind2) ...
          + (1-eta(:,1)).*eta(:,2)     .* pdf_dgx(linind3) ...
          + eta(:,1).*eta(:,2)         .* pdf_dgx(linind4);
    dpy = (1-eta(:,1)).*(1-eta(:,2)) .* pdf_dgy(linind) ...
          +  eta(:,1).*(1-eta(:,2))    .* pdf_dgy(linind2) ...
          + (1-eta(:,1)).*eta(:,2)     .* pdf_dgy(linind3) ...
          + eta(:,1).*eta(:,2)         .* pdf_dgy(linind4);
    
%     dH_binned = cat(3,  imfilter(dpsi_binned,dG(:,:,1),'same'), ...
%                         imfilter(dpsi_binned,dG(:,:,2),'same'));
	dH_binnedx = imfilter(dpsi_binned,dG(:,:,1),'same');
    dH_binnedy = imfilter(dpsi_binned,dG(:,:,2),'same');
%         
%     dH = zeros(N,2);
%     for n=1:N
%         dH(n,:) = dp(n,:)' * dpsi(n) + ...
%                 (1-eta(n,1))*  (1-eta(n,2)) * reshape(dH_binned(mhash(n,1),  mhash(n,2),:),2,1) + ...
%                  eta(n,1)   *  (1-eta(n,2)) * reshape(dH_binned(mhash(n,1)+1,mhash(n,2),:),2,1) + ...
%                 (1-eta(n,1))*   eta(n,2)    * reshape(dH_binned(mhash(n,1),  mhash(n,2)+1,:),2,1) + ...
%                  eta(n,1)   *   eta(n,2)    * reshape(dH_binned(mhash(n,1)+1,mhash(n,2)+1,:),2,1);
%     end
%     dH = -dH;   % This sign is different compared to fastentropy1d (due to the binning choices probably)
    dHx = dpx .* dpsi + (1-eta(:,1)).*(1-eta(:,2)) .* dH_binnedx(linind) ...
                      +  eta(:,1).*(1-eta(:,2))    .* dH_binnedx(linind2) ...
                      + (1-eta(:,1)).*eta(:,2)     .* dH_binnedx(linind3) ...
                      + eta(:,1).*eta(:,2)         .* dH_binnedx(linind4);
	dHy = dpy .* dpsi + (1-eta(:,1)).*(1-eta(:,2)) .* dH_binnedy(linind) ...
                      +  eta(:,1).*(1-eta(:,2))    .* dH_binnedy(linind2) ...
                      + (1-eta(:,1)).*eta(:,2)     .* dH_binnedy(linind3) ...
                      + eta(:,1).*eta(:,2)         .* dH_binnedy(linind4);
	dH = -[dHx dHy];
end



% Same calculation without for-loop (approx. same time as above)
% linind = sub2ind([nbins,nbins],  mhash(:,1)   ,mhash(:,2));
% linind2 = sub2ind([nbins,nbins], mhash(:,1)+1 ,mhash(:,2));
% linind3 = sub2ind([nbins,nbins], mhash(:,1)   ,mhash(:,2)+1);
% linind4 = sub2ind([nbins,nbins], mhash(:,1)+1 ,mhash(:,2)+1);
% p2 =    (1-eta(:,1)).*(1-eta(:,2)) .* pdf_g(linind) ...
%       +  eta(:,1).*(1-eta(:,2))    .* pdf_g(linind2) ...
%       + (1-eta(:,1)).*eta(:,2)     .* pdf_g(linind3) ...
%       + eta(:,1).*eta(:,2)         .* pdf_g(linind4);
% H2 = -1/N * sum(log(p2(p2>0)))
