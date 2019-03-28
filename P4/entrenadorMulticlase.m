function theta = entrenadorMulticlase(X, y, totalNumeros, lambda)

% Numero de ejemplos
m = size(X, 1);
% Numero de parametros
n = size(X, 2);

% Inicializamos la matriz solucion 'theta'
theta = zeros(totalNumeros, n+1);
theta_inicial = zeros(n+1, 1);
X = [ones(m, 1) X];


warning off;

% Entrenamos para cada clase. 
% Rellenamos la matriz de clasificadores one-vs-all
for i=1:totalNumeros
    
    % Seleccionamos solo las muestras de la clase 'i'
    yTemporal = (y==i);
    
    % Obtenemos 'thetaTemporal' mediante regularizacion logistica
    options = [];
    options.display = 'none';
    [thetaTemporal(:,i)] = minFunc(@CosteLogReg,theta_inicial,options,X,yTemporal,lambda);
    
    % Añadimos la theta obtenida a la matriz de clasificadores
    theta(i,:) = thetaTemporal(:,i)';
end