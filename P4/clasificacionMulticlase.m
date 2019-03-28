
function p = clasificacionMulticlase(theta, X)

m = size(X, 1);
p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];

% Calculamos las predicciones
%predicciones = theta * X';

predicciones = 1./(1+exp(-(theta * X')));

% Seleccionamos la prediccion que mas se ajusta de cada clase
[maxval, maxindice] = max(predicciones);
p = maxindice';