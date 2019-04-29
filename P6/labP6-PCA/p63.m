% load images 
% images size is 20x20. 
clear
close all

load('MNISTdata2.mat'); 

nrows=20;
ncols=20;

nimages = size(X,1);


%% Perform PCA over all numbers
Xest = Xest - repmat(mean(X), size(X, 1), 1);
mCov = cov(Xest);
[U, V] = eig(mCov);

%% Perform PCA over all numbers
Xtest = Xtest - repmat(mean(X), size(Xtest, 1), 1);

% Ordenacion de las dos primeras componentes
vDiagAscend = diag(sort(diag(V)','descend'));
vDiag = sort(diag(V)','descend');
vDiagRef = diag(V)';

% La matriz de los vectores propios ordenada de mayor a menor
% segun su valor propio es la siguiente
[~, I] = sort(vDiagRef, 'descend');
Uord = U(:,I');

% La matriz Uord ya esta ordenada
Ureduce = Uord(1:2);
Z = Ureduce' * Xest';

% Muestra las dos componentes principales
figure()
clf, hold on
plotwithcolor(Z(:,1:2), y);

% Seleccion de las clases que mejor se clasifican en entrenamiento
matrix = find((y == 10) | (y == 1));
Yok = y(matrix,:);
Xok = X(matrix,:);

% Seleccion de las clases que mas se confunden en entrenamiento
matrix = find((y == 8) | (y == 3));
Ynot = y(matrix,:);
Xnot = X(matrix,:);

% Seleccion de las clases que mejor se clasifican en test
matrix = find((y == 10) | (y == 1));
Yoktest = y(matrix,:);
Xoktest = Xtest(matrix,:);

% Seleccion de las clases que mas se confunden en test
matrix = find((y == 8) | (y == 3));
YnotTest = y(matrix,:);
XnotTest = Xtest(matrix,:);

% La matriz Uord ya esta ordenada
Ureduce = Uord(1:2);

% Obtencion de la matriz Z de entrenamiento y de test
Z = Ureduce' * Xest';
Ztest = Ureduce' * Xtest';

%% Use classifier from previous labs on the projected space
rand('state',0);
p = randperm(length(y));
Z = Z(p,:);
y = y(p);

% Calculo del mejor valor de lambda
[mejor_lambda] = validacionCruzada(Z, y, 10, 0);
modelo = entrenarGaussianas(Z, y, 10, 0, mejor_lambda);

% Obtencion de la prediccion
prediccion = clasificacionBayesiana(modelo, Z);











