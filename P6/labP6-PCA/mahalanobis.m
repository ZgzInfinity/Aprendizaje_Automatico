function [d2, R] = mahalanobis(mu, Sigma, X)
% Calcula las distancias de Mahalanobis de las muestras X a
% la Gaussiana N(mu, Sigma). Tambien devuelve R = chol(Sigma)

%X_menos_mu = bsxfun(@minus, X, mu');
X_menos_mu = X - repmat(mu',size(X,1),1);
R = chol(Sigma);
z  = X_menos_mu / R;
d2 = sum(z.^2,2);  %Distancia de Mahalanobis