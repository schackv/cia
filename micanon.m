function [micv1,micv2,V1,V2, mi, I, Inorm, cca_out, optim_out] =  micanon(X,Y,varargin)
% MICANON  Performs canonical information analysis (CIA) on two sets of
% multivariate data. The goal is to maximize mutual information between the
% two linear combinations U = a^T X and V = b^T Y
% This is useful for data sets where e.g. correlation does not make sense
% as a measure for similarity.
% Only the leading pair of mutual information canonical variates (MICV) is
% determined.
% 
% The data is sphered prior to analysis.
%
% Example:
%   Consider one signal y = x and another signal y2 = x.^2. 
%   Let y be the first variable of the first set of variables 
%   and random noise the second. Let y2 be the first
%   variable of the second set of variables and random noise the second.
%   When sampled between 0 and 1, correlation makes perfect sense as a
%   measure for similarity and yields the obvious solution a = [1;0], b=[1;0]. 
%   But when sampled between -1 and 1, this breaks down and 
%   the solution yielded by canonical correlation analysis (a=[], b=[]) 
%   is meaningless. In both these cases CIA yields approximately 
%   a=[1;0], [1;0] as expected.
%     Nsamples = 1000;
%     xx = linspace(-1,1,Nsamples)';      % Change -1 to 0 for case where
%                                           CCA makes sense
%     X1 = xx.^2 + 0.1*randn(Nsamples,1);
%     Y1 = xx;
%     X2 = 0.1*randn(Nsamples,1);
%     Y2 = 0.1*randn(Nsamples,1);
%     X = [X1, X2];
%     Y = [Y1, Y2];
%     [micv1,micv2,v1,v2,mi,Inorm,cca_out,optimout] = micanon(X,Y);
%     disp('CIA solution:');
%     disp(v1); disp(v2);
%     disp('CCA solution:');
%     disp(cca_out.A(:,1)); disp(cca_out.B(:,1));
% 
%
% Inputs
%   X           First set of multivariate data of size N x p1
%   Y           Second set of multivariate data N x p2
%
%   Optional inputs as parameter/value pairs
%   'xstart'            Starting point for optimization algorithm. Input 
%                       A vector [a0;b0] of size p1+p2 to specify starting point
%                       'cca' to start at CCA optimum, or
%                       'equal' to start with equally weighted variables (default)
%   'presphere'         Sphere data prior to optimization (default=true)
%   'bandwidth'         Bandwidth estimation method
%           'MSP'       Maximal smoothing principle (default)
%           'Silverman' Silverman's rule of thumb
%   'method'            Optimization function handle. Function should have
%                       the signature:
%           [x,y,funevals] = optimfun( @minfun, xstart, args)
%
%                       Wrappers for a couple of common matlab
%                       optimizers are included.
%                       'fminsearch_wrapper':   fminsearch
%                       'simanneal_wrapper':    simanneal
%                       'genetic_wrapper':      ga
%                       'bfgs_wrapper':         bfgs
%
%                       where @minfun is the function to be minimized.
%   'numcomp'           Number of components. Default = 1.
%   'spherical'         True to optimize on two hyperspheres (default)
%                       False to optimize in standard cartesian coordinates
%                       Optimizing in spherical coordinates effectively reduces
%                       the number of dimensions with two, one for each set of
%                       variables.
%   'display'           'on' or 'off' to show optimization progress or not.
%
% Outputs
%   micv1               Leading mutual information canonical variate (N x 1)
%                       corresponding to first set of variables
%   micv2               Leading mutual information canonical variate (N x 1)
%                       corresponding to second set of variables
%   v1                  Projection direction for first set of variables (p1-vector)
%   v2                  Projection direction for second set of variables (p2-vector)
%   mi                  Vector of normed mutual information between each of the 
%                       input variables and the corresponding and MICV (p1+p2 x 1)
%   I                   Mutual information 
%   Inorm               Normalized mutual information between the two 
%                       resulting MICVs (scalar)
%   optim_out           Struct containing optimization related output
%       .fval               Final optimization value (
%       .feval              Number of function evaluations
%       .history            Optimization history
%       .xstart             Starting point for optimization
%   cca_out             Struct containing CCA related output (see canoncorr)
%       .A                  Canonical coefficients for X
%       .B                  Canonical coefficients for Y
%       .r                  Canonical correlations for CCA solution
%
% References
%   Vestergaard, J. S., Nielsen, A. A., 
%   Canonical Information Analysis (CIA)
%   http://www.imm.dtu.dk/pubdb/p.php?6270

% Jacob S. Vestergaard
% jsve@imm.dtu.dk


import cia.*;

%% Initialize
X = zscore(X);
Y = zscore(Y);

[N , nvar ] = size(X);
[N2, nvar2] = size(Y);

if N~=N2
    error('CIA:NumberOfSamples','The number of samples in each set must be equal');
end

% History variables
history.x = [];
history.fval = [];
history.p = [];

% Parse options
args = struct('x0', 'equal', ...
                'fastentropy', true,...
                'usegradient',false,...
              'presphere',true,...
              'bandwidth','MSP', ...
             'method', 'fminsearch_wrapper', ...
             'spherical', true, ...
             'display','on', ...
             'numcomp',1);

args = parseArgs(varargin, args);

if strcmp(args.display,'on')
          args.PlotFcns = ...
                {@optimplotfval, ...
                @optimplotx,...
                @optimplotstepsize};
else
    args.PlotFcns = {};
end
args.OutputFcn = @outfun;

% Initialize output arguments
    
mi = NaN(nvar+nvar2,args.numcomp);
I  = NaN(args.numcomp,1);   
Inorm=I;
V1 = NaN(nvar,args.numcomp);
V2 = NaN(nvar2,args.numcomp);
fval = NaN(args.numcomp,1);
funeval = NaN(args.numcomp,1);

%% Do CCA for reference
tic;
[cca_out.A, cca_out.B, cca_out.r,cca_out.U,cca_out.V] = canoncorr(X, Y);
cca_out.T = toc;

%% Center data + whitening
if args.presphere, 
    [X,Tx] = sphere(X);
    [Y,Ty] = sphere(Y);
end

%% Determine components sequentially
Xhat = X;   % Start with original (perhaps sphered) data
Yhat = Y;
for p=1:args.numcomp
    optimargs = {'spherical',args.spherical, ...
                   'V1',V1, ...
                   'V2',V2, ...
                   'p', p, ...
                   'bandwidth',args.bandwidth, ...
                   'fastentropy', args.fastentropy, ...
                   'usegradient', args.usegradient};
    
    %% Initialize as CCA optimum if empty start criterion
    if strcmpi(args.x0, 'cca')
        % Will initialize as CCA optimum
        xstart = [cca_out.A(:,p)
                  cca_out.B(:,p)];
    elseif strcmpi(args.x0,'equal')
        xstart = [sqrt(nvar)/nvar*ones(1,nvar), ...
                        sqrt(nvar2)/nvar2*ones(1,nvar2)]';
    else
        xstart = args.x0;
    end

    %% Convert start values to spherical coordinates if chosen
    optim_out.xstart(:,p) = xstart;
    if args.spherical
        [r1,a] = cart2hypersph(xstart(1:nvar)');
        if nvar2==1 % Second set is univariate
            b = [];
        else
            [r2,b] = cart2hypersph(xstart(nvar+1:end)');
        end
        xstart = [a, b]';
        nvar = nvar-1;
        nvar2 = nvar2-1;
    end

    %% Optimize
    disp('****** Maximum mutual information ******')
    mifun = @(x) maxMIs(x,nvar,Xhat, Yhat, optimargs{:});
    [xhat, fval(p), funeval(p)] = feval(args.method,mifun,xstart,args);

    %% Transform spherical coordinates back to cartesian
    xhat = xhat(:);
    switch args.spherical
        case true
            xhat = [hypersph2cart(1, xhat(1:nvar)'), hypersph2cart(1, xhat(nvar+1:end)')]';
            nvar = nvar+1;
            nvar2 = nvar2+1;
    end

    % Add to already found components
    V1(:,p) = xhat(1:nvar)/norm(xhat(1:nvar));
    V2(:,p) = xhat(nvar+1:end)/norm(xhat(nvar+1:end));
     
%     micv1(:,p) = Xhat*V1(:,p);    % 'Adjusted' components
%     micv2(:,p) = Yhat*V2(:,p);
    
    % Do structure removal
    Xhat = remove_structure(Xhat,V1(:,p));
    Yhat = remove_structure(Yhat,V2(:,p));
    
    
end

% MICVs
micv1 = X*V1;
micv2 = Y*V2;

for i=1:args.numcomp
    % Calculate MI between MICV pairs
    [I(i),~,Inorm(i)]         = kdeMI(micv1(:,i),micv2(:,i),'bandwidth',args.bandwidth);
    if i<=size(cca_out.U,2), [cca_out.I(i),~,cca_out.Inorm(i)]         = kdeMI(cca_out.U(:,i),cca_out.V(:,i),'bandwidth',args.bandwidth); end
    % Calculate MI between input bands and MICVs
    for j=1:nvar
        mi(j,i)         = kdeMI(X(:,j),micv1(:,i),'bandwidth',args.bandwidth);
    end
    for j=1:nvar2
        mi(j+nvar,i)    = kdeMI(Y(:,j),micv2(:,i),'bandwidth',args.bandwidth);
    end
end

% Return vectors directly applicable to standardized data
if args.presphere  
    V1 = Tx*V1;
    V2 = Ty*V2;
end

% Prepare other output
if ~isempty(history.x) && args.spherical
    history.x = [hypersph2cart(1, history.x(:,1:nvar)), ...
                hypersph2cart(1, history.x(:,nvar+1:end))];
end

I = -fval;  % Mutual informations between p'th pair
optim_out.fval = fval;     
optim_out.feval = funeval;
optim_out.history = history;



%% Internal util functions
    % Output function - store iteration history
    function stop = outfun(x,optimValues,state,varargin)
         stop = false;

         switch state
             case 'init'
    %              hold on
             case 'iter'
             % Concatenate current point and objective function
             % value with history. x must be a row vector.
               history.fval = [history.fval; optimValues.fval];
               history.x = [history.x; x(:)'];
               history.p = [history.p; p];
             case 'done'
    %              hold off
             otherwise
         end
    end


    % Variance constraints (only used for fmincon)
    function [c, ceq, Gc, Gceq] = ...
        varcon(x, n1, X, Y, s11, s22,varargin)

        inargs = struct('coordinates','cart');
        inargs = parseArgs(varargin, inargs);

        x = x(:);
        if inargs.spherical==true
            x = [hypersph2cart(1, x(1:n1)'), hypersph2cart(1,x(n1+1:end)')]';
            n1 = n1+1;
        end

        n = length(x);
        % constraints
        c = [];
        ceq = [x(1:n1)'*s11*x(1:n1) - 1; x(n1+1:end)'*s22*x(n1+1:end) - 1];

        % gradients of constraints
            Gc = [];
            Gceq = 2*[s11*x(1:n1) zeros(n1,1); zeros(n-n1,1) s22*x(n1+1:end)];
    end
end


% function [y, grad, Hxy, Hx, Hy,bandwidth] = maxMIs(x, nvar1,X, Y, varargin)
function [y, grad] = maxMIs(x, nvar1,X, Y, varargin)
% MAXMI Calculates the negative mutual information of X and Y given the
% projections in x. MI is estimated using kernel density estimation.

    args = struct('spherical',true,'bandwidth',[],'V1',[],'V2',[],'p',1,'fastentropy',true,'usegradient',false);
    args = parseArgs(varargin, args);

    x = x(:);
    a = x(1:nvar1);     % Extract first projection direction
    b = x(nvar1+1:end); % Extract second projection direction

    if args.spherical
        % Convert to cartesian coordinates
        a = hypersph2cart(1, a')';
        b = hypersph2cart(1, b')';
    end

    CV1 = X*a;
    CV2 = Y*b;

    [Iout, dIdxy] = kdeMI(CV1,CV2,'bandwidth',args.bandwidth,'fastentropy',args.fastentropy,'usegradient',args.usegradient);
    fval = Iout;

    if ~isempty(dIdxy)
        N=size(X,1);
        % Convert to function of a and b
        dIda = X'*dIdxy(:,1);
        dIdb = Y'*dIdxy(:,2);
        if args.spherical
            % Convert gradient back to cartesian basis
            error('Back convertion from spherical coordinates not implemented')
        end
        grad = -[dIda; dIdb];
    end
    y = -fval;

      
    

end

function [f, J, bandwidth] = maxMI_grad(x, nvar1, X, Y, varargin)
    args = struct('coordinates','cart','bandwidth',[],'Dx',[],'Dy',[]);
    args = parseArgs(varargin, args);

    x = x(:);
    a = x(1:nvar1);     % Extract first projection direction
    b = x(nvar1+1:end); % Extract second projection direction
       
    bandwidth = args.bandwidth;
    if isempty(bandwidth)
        bandwidth = ksizeMSP([X*a Y*b]');
    end
    
    switch args.spherical
        case true
            % Convert to cartesian coordinates
            a = hypersph2cart(1, a')';
            b = hypersph2cart(1, b')';
    end
    
    
    [I, dI] = differentialMI(X,Y,a,b,'bandwidth',args.bandwidth, ...
                                     'Dx', args.Dx,'Dy',args.Dy);
    
    f = -I;
    J = -dI;
end


function [xval,fval,funevals] = fminsearch_wrapper(fun,xstart,args)
    options = optimset('fminsearch');
    options.OutputFcn = args.OutputFcn;
    options.PlotFcns = args.PlotFcns;
    options.TolFun = 1e-4;
    options.TolX = 1e-4;
%     options.MaxIter = 300;
%     warning('Setting max. iterations to 300!');
    [xval, fval,exitflag,options] = fminsearch(fun, xstart,options);
    funevals = options.funcCount;
end

function [xval,fval,funevals] = bfgs_wrapper(fun,xstart,args)
    options = optimoptions('fminunc');
    options.OutputFcn = args.OutputFcn;
    options.PlotFcns = args.PlotFcns;
    
    if args.usegradient, 
        options.GradObj='on';
%         options.DerivativeCheck='on';
    else
        options.GradObj = 'off';
    end
    [xval, fval,exitflag,options] = fminunc(fun, xstart,options);
    funevals = options.funcCount;
end

function [xval,fval,funevals] = simanneal_wrapper(fun,xstart,args)
    options = saoptimset('simulannealbnd');    
    options.OutputFcn = args.OutputFcn;
    if strcmp(args.display,'on')
        options.PlotFcns = {@saplotbestx, @saplotbestf, @saplotx, @saplotf};
    end
    [xval,fval,exitflag,output] = simulannealbnd(fun,xstart,[],[],options);
    funevals = output.funccount;
end

function [xval,fval,funevals] = genetic_wrapper(fun,xstart,args)
    options = gaoptimset(@ga);
    options.PopulationSize = 5*length(xstart)^2;
%     options.OutputFcns = args.OutputFcn;
    if strcmp(args.display,'on')
        options.PlotFcns = {@gaplotbestf , @gaplotbestindiv , @gaplotdistance , @gaplotexpectation , @gaplotgenealogy , @gaplotmaxconstr , @gaplotrange , @gaplotselection , @gaplotscorediversity , @gaplotscores , @gaplotstopping };
    end
    [xval,fval,exitflag,output] = ga(fun,length(xstart),[],[],[],[],[],[],[],[],options);
    funevals = output.funccount;
end





%% Variance constraints
function [c, ceq, Gc, Gceq] = ...
    varcon(x, n1, X, Y, s11, s22,varargin)

inargs = struct('coordinates','cart');
inargs = parseArgs(varargin, inargs);

x = x(:);
if inargs.spherical==true
    x = [hypersph2cart(1, x(1:n1)'), hypersph2cart(1,x(n1+1:end)')]';
    n1 = n1+1;
end

n = length(x);
% constraints
c = [];
%ceq = [x(:,1)'*s11*x(:,1) - 1; x(:,2)'*s22*x(:,2) - 1];
ceq = [x(1:n1)'*s11*x(1:n1) - 1; x(n1+1:end)'*s22*x(n1+1:end) - 1];

% gradients of constraints
%if nargout > 2
    Gc = [];
    %Gceq = 2*[s11*x(:,1); s22*x(:,2)];
    Gceq = 2*[s11*x(1:n1) zeros(n1,1); zeros(n-n1,1) s22*x(n1+1:end)];
%end
end