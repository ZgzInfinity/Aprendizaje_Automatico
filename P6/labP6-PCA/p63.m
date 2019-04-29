% load images 
% images size is 20x20. 
clear
close all

load('MNISTdata2.mat'); 

nrows=20;
ncols=20;

nimages = size(X,1);

%Perform PCA over all numbers
Xest = X - repmat(mean(X), size(X, 1), 1);
mCov = cov(Xest);
[U, V] = eig(mCov);

% Centrado de los datos de test
XtestEst = Xtest - repmat(mean(Xtest), size(Xtest, 1), 1);


% Ordenacion de las dos primeras componentes
vDiagAscend = diag(sort(diag(V)','descend'));
vDiag = sort(diag(V)','descend');
vDiagRef = diag(V)';

% La matriz de los vectores propios ordenada de mayor a menor
% segun su valor propio es la siguiente
[~, I] = sort(vDiagRef, 'descend');
Uord = U(:,I');

% Obtencion de la matriz Z de entrenamiento
Ureduce = Uord(:,1:2);
Z = Ureduce' * Xest';
Ztest = Ureduce' * XtestEst';

% Muestra las dos componentes principales
figure()
clf, hold on
title('Grafico con las dos primeras componentes de cada imagen');
plotwithcolor(Z'(:,1:2), y);

% Trasposicion de las matrices
Z = Z';
Ztest = Ztest';

% Seleccion de las clases que mas se confunden en entrenamiento
matrix = find((y == 10) | (y == 1));
Yok = y(matrix,:);
Zok = Z(matrix,:);

% Seleccion de las clases que mas se confunden en entrenamiento
matrix = find((y == 8) | (y == 3));
Ynot = y(matrix,:);
Znot = Z(matrix,:);

% Seleccion de las clases que mejor se clasifican en test
matrix = find((ytest == 10) | (ytest == 1));
Yoktest = ytest(matrix,:);
Zoktest = Ztest(matrix,:);

% Seleccion de las clases que mas se confunden en test
matrix = find((ytest == 8) | (ytest == 3));
YnotTest = ytest(matrix,:);
ZnotTest = Ztest(matrix,:);


%% Use classifier from previous labs on the projected space
rand('state',0);
p = randperm(length(Yok));
Zok = Zok(p,:);
Yok = Yok(p,:);


%% MEJORES VALORES 10 Y 1 %%

% Calculo del mejor valor de lambda para cada clasificador
[mejor_lambda] = validacionCruzada(Zok, Yok, 0, 1, 10);
modelo = entrenarGaussianas(Zok, Yok, 0, mejor_lambda, 1, 10);

% Obtencion de la prediccion para cada clasificador
prediccion = clasificacionBayesiana(modelo, Zoktest, 1, 10);

error_test = ((1 - (mean(double(prediccion == Yoktest)))) * 100);
fprintf('Error con datos de test = %f\n', error_test);

% Matrices de confusion para cada valor mejor
matrizConfusion(prediccion, Yoktest, 1, 10);

%% PEORES VALORES 8 Y 3 %%

% Calculo del mejor valor de lambda para cada clasificador
[mejor_lambda] = validacionCruzada(Znot, Ynot, 0, 3, 8);
modelo = entrenarGaussianas(Znot, Ynot, 0, mejor_lambda, 3, 8);

% Obtencion de la prediccion para cada clasificador
prediccion = clasificacionBayesiana(modelo, ZnotTest, 3, 8);

error_test = ((1 - (mean(double(prediccion == YnotTest)))) * 100);
fprintf('Error con datos de test = %f\n', error_test);

% Matrices de confusion para cada valor mejor
matrizConfusion(prediccion, YnotTest, 8, 3);


fprintf('A continuacion es el ejercicio 2\n\n\n');


%% PARTE 2 DE LA PRACTTICA 

% Valores de entrenamiento
matrix = find((y == 10) | (y == 1));
Yok = y(matrix,:);
Xok = X(matrix,:);

% Centrar los datos de entrenamiento
Xest = Xok - repmat(mean(Xok), size(Xok, 1), 1);

% Valores de test
matrix = find((y == 10) | (y == 1));
YokTest = ytest(matrix,:);
XokTest = Xtest(matrix,:);

% Centrar los datos de Test
XtestEst = XokTest - repmat(mean(Xok), size(XokTest, 1), 1);

% Matriz de covarianza
mCov = cov(Xest);
[U, V] = eig(mCov);

% Ordenacion de las dos primeras componentes
vDiagAscend = diag(sort(diag(V)','descend'));
vDiag = sort(diag(V)','descend');
vDiagRef = diag(V)';

% La matriz de los vectores propios ordenada de mayor a menor
% segun su valor propio es la siguiente
[~, I] = sort(vDiagRef, 'descend');
Uord = U(:,I');

% Obtencion de la matriz Z de entrenamiento
Ureduce = Uord(:,1:2);
Z = Ureduce' * Xest';
Ztest = Ureduce' * XtestEst';

% Trasposicion de las Z 
Z = Z';
Ztest = Ztest';

%% Permutacion de los datos de Z
rand('state',0);
p = randperm(length(Yok));
Z = Z(p,:);
Yok = Yok(p,:);

%% VALORES 10 Y 1 PERO CON PCA PARA LOS K = 2

% Calculo del mejor valor de lambda para cada clasificador
[mejor_lambda] = validacionCruzada(Z, Yok, 0, 1, 10);
modelo = entrenarGaussianas(Z, Yok, 0, mejor_lambda, 1, 10);

% Obtencion de la prediccion para cada clasificador
prediccion = clasificacionBayesiana(modelo, Ztest, 1, 10);

error_test = ((1 - (mean(double(prediccion == YokTest)))) * 100);
fprintf('Error con datos de test = %f\n', error_test);

% Matrices de confusion para cada valor mejor
matrizConfusion(prediccion, YokTest, 1, 10);

fprintf('Vista de la matriz de confusion global\n');
matrizConfusionGorda(prediccion, YokTest, 1, 10);

%% VALORES 8 Y 3 PERO CON PCA PARA LOS K = 2


matrix = find((y == 8) | (y == 3));
Ynot = y(matrix,:);
Xnot = X(matrix,:);

% Centrar los datos de entrenamiento
Xest = Xnot - repmat(mean(Xnot), size(Xnot, 1), 1);

% Valores de test
matrix = find((ytest == 8) | (ytest == 3));
YnotTest = ytest(matrix,:);
XnotTest = Xtest(matrix,:);

% Centrar los datos de Test
XtestEst = XnotTest - repmat(mean(Xnot), size(XnotTest, 1), 1);

% Matriz de covarianza
mCov = cov(Xest);
[U, V] = eig(mCov);

% Ordenacion de las dos primeras componentes
vDiagAscend = diag(sort(diag(V)','descend'));
vDiag = sort(diag(V)','descend');
vDiagRef = diag(V)';

% La matriz de los vectores propios ordenada de mayor a menor
% segun su valor propio es la siguiente
[~, I] = sort(vDiagRef, 'descend');
Uord = U(:,I');

% Obtencion de la matriz Z de entrenamiento
Ureduce = Uord(:,1:2);
Z = Ureduce' * Xest';
Ztest = Ureduce' * XtestEst';

% Trasposicion de las Z 
Z = Z';
Ztest = Ztest';

%% Permutacion de los datos de Z
rand('state',0);
p = randperm(length(Yok));
Z = Z(p,:);
Ynot = Ynot(p,:);


% Calculo del mejor valor de lambda para cada clasificador
[mejor_lambda] = validacionCruzada(Z, Ynot, 0, 3, 8);
modelo = entrenarGaussianas(Z, Ynot, 0, mejor_lambda, 3, 8);

% Obtencion de la prediccion para cada clasificador
prediccion = clasificacionBayesiana(modelo, Ztest, 3, 8);

error_test = ((1 - (mean(double(prediccion == YnotTest)))) * 100);
fprintf('Error con datos de test = %f\n', error_test);

% Matrices de confusion para cada valor mejor
matrizConfusion(prediccion, YnotTest, 3, 8);


fprintf('Vista de la matriz de confusion global\n');
matrizConfusionGorda(prediccion, YnotTest, 3, 8);
















