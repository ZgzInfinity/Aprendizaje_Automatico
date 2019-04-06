function x=myMvnRnd(mu, Sigma, nsamples)
% INPUT:
%    mu (nx1): media
%    Sigma (n,n): matriz de covarianza
%    nsamples: numero de muestras 
% OUTPUT:
%    x(n x nsamples): muestras de una Gausiana de media mu y covarianza
%    Sigma

[R,p]=chol(Sigma);

x0=randn(length(mu), nsamples);
x=repmat(mu,1,nsamples) +R'*x0;