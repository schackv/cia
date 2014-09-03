
s = RandStream('mt19937ar','Seed',143);
RandStream.setGlobalStream(s);

%% Simulate toy example data
Nsamples = 1e4;
sigma = 0.1;
x1 = rand(Nsamples,1)*2 - 1;            % Sample between -1 and 1
x2 = 0.1*randn(Nsamples,1);             % Gaussian noise
y1 = x1.^2 + sigma*randn(Nsamples,1);   % Make y1 a nonlinear function of x1 and add noise with st.d.=sigma
y2 = 0.1*randn(Nsamples,1);             % Gaussian noise
X = [x1, x2];
Y = [y1, y2];

%% Canonical correlation analysis
[A_cca, B_cca,~, ccacv1, ccacv2 ,stats_cca] = canoncorr(X,Y);


%% Canonical information analysis
[micv1,micv2,A_cia, B_cia, mi, I, Inorm, cca_out, stats_cia]  = micanon(X,Y,'method','simanneal_wrapper','numcomp',1,'spherical',false);

% Also try with one of the other 'method's:
% 'fminsearch_wrapper','bfgs_wrapper','simanneal_wrapper','genetic_wrapper'

%% Show scatter plots for the first set for each method
figure;
subplot 121;
plot(ccacv1(:,1),ccacv2(:,1),'.b');
title('Canonical correlation analysis');

subplot 122;
plot(micv1(:,1),micv2(:,1),'.b');
title('Canonical information analysis');
