function [I, dI, Inorm, Hxy, Hx, Hy, bandwidth] = kdeMI(x, y, varargin)
% [I Inorm Hxy Hx Hy] = kdeMI(x, y):
%
% kdeMI calculates mutual information I(x,y) using kernel density estimation
% of the joint pdf
%
%   Inputs
%       x       First variable [N x 1]
%       y       Second variable [N x 1]
%
%   Outputs
%       I       Mutual information I(x,y) = H(x) + H(y) - H(x,y)
%       Inorm   Normed mutual information 2*I(x,y)/(H(x) + H(y)) also known
%               as redundancy
 
args = struct('fastentropy',true,'usegradient',false,'bandwidth','MSP','entropy','shannon','nbins',128);
args = parseArgs(varargin,args);

x = x(:);
y = y(:);

% Estimate bandwidth using Maximal Smoothing Principle
if ischar(args.bandwidth)
    switch args.bandwidth
        case 'MSP'
            bandwidth.x  = ksizeMSP(x');
            bandwidth.y  = ksizeMSP(y');
            bandwidth.xy = ksizeMSP([x y]');
%             bandwidth.xy = [bandwidth.x, bandwidth.y];  % Use same bandwidth to ensure that p(x,x)-p(x)p(x)=0
        case 'Silverman'    % The choice of Yin 2004
            s1 = std(x);   
            s2 = std(y);   
            bandwidth.x = 1.06*s1*n^(-0.2);
            bandwidth.y = 1.06*s2*n^(-0.2);
            bandwidth.xy = [s1 s2].*n^(-1/6);
    end
else
    bandwidth.x  = args.bandwidth(1);
    bandwidth.y  = args.bandwidth(2);
    bandwidth.xy = args.bandwidth;
end

if args.fastentropy
    fun1d = @(x,bandwidth)fastentropy1d(x,bandwidth,args.nbins);
    fun2d = @(x,y,bandwidth)fastentropy2d([x y],bandwidth,args.nbins);
else
    fun1d = @(x,bandwidth)slowentropy1d(x,bandwidth);
    fun2d = @(x,y,bandwidth)slowentropy2d(x,y,diag(bandwidth.^2));
end

switch args.entropy
    case 'shannon'
        if ~args.usegradient
            [~,Hx] = feval(fun1d,x,bandwidth.x);
            [~,Hy] = feval(fun1d,y,bandwidth.y);
            [~,Hxy] = feval(fun2d,x,y,bandwidth.xy);
            dI = [];
        else
            [~,Hx,dHx] = feval(fun1d,x,bandwidth.x);
            [~,Hy,dHy] = feval(fun1d,y,bandwidth.y);
            [~,Hxy,dHxy] = feval(fun2d,x,y,bandwidth.xy);
            dI = [dHx dHy]-dHxy;     % [dIdx dIdy]
        end


        % Calculate MI
        I = Hx+Hy-Hxy;
        if nargout>2
            [~,Hxx] = feval(fun2d,x,x,[bandwidth.x bandwidth.x]);
            [~,Hyy] = feval(fun2d,y,y,[bandwidth.y bandwidth.y]);
            Inorm = I/Hxy;
        end
    case 'renyi'
        if args.usegradient
            error('cia:NotImplemented','Gradient calculation not implemented for Renyi entropy.');
        end
        px = feval(fun1d,x,bandwidth.x);
        py = feval(fun1d,y,bandwidth.y);
        pxy = feval(fun2d,x,y,bandwidth.xy);
        N = length(x);
        I = -log(N) + log(sum(pxy./px./py));
        dI = [];
end



    
end

function nmi = normed_differential_mi(n,h_x,h_y,h_xy,xrange,yrange,k)
    % NORMED_DIFFERENTIAL_MI Estimate normalized mutual information based
    % on differential entropy estimates. This is similar to a conversion
    % from differential to discrete entropy
    % Use the fact that H(x) = h(x) - log(delta)
    
    % Estimate number of bins (Sturge's formula)
%     k = log2(n) + 1;
%     k = 128;
    delta_x = diff(xrange)./k;
    delta_y = diff(yrange)./k;
    nmi = (h_x + h_y - h_xy)/(h_x + h_y - log(delta_x*delta_y));
end