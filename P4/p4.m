%%
%% Practica 4 - Fernando Aliaga Ramon
%%

%
% Ejercicio 2 - Regresion logistica regularizada
%
fprintf('\nEjercicio 2 - Regresion logistica regularizada\n');
fprintf('----------------------------------------------\n');

clear ; close all;
addpath(genpath('../minfunc'));

% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);

% Resolvemos la regresion logistica regularizada para encontrar el
% parametro landa
mejor_lambda = validacionCruzada (0,0.01,0.001,X,y,10);

%
% Ejercicio 3 - Matriz de confusion y precision/recall
%
fprintf('Ejercicio 3 - Matriz de confusion y precision/recall\n');
fprintf('----------------------------------------------------\n');

% Obtenemos theta para el clasificador multiclase
theta = entrenadorMulticlase(X, y, 10, mejor_lambda);

% Calculamos los errores.
% Error con datos de entrenamiento
p = clasificacionMulticlase(theta, Xtest);
error_test = ((1 - (mean(double(p == ytest))))*100);
fprintf('Error con datos de test = %f\n',error_test);

% Calculamos la matriz de confusion para cada clase
for i=1:10
    matrizConfusion(p,ytest,i);
end

% Inventa una solucion y muestra las confusiones
verConfusiones(Xtest, ytest, p);

matriz = [ytest p];
