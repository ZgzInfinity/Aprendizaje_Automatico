%%
%% Practica 4 - Ruben Rodriguez Esteban
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
mejor_lambda = validacionCruzada(X,y,10);
fprintf('El mejor Lambda hallado es %f\n', mejor_lambda);

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


fprintf('Matrices de confusion para cada clasificador\n');
fprintf('Formato de la matriz de confusion\n');
fprintf('TP  FP\n');
fprintf('FN  TN\n');
fprintf('TP => true positive\n');
fprintf('FP => false positive\n');
fprintf('FN => false negative\n');
fprintf('TN => true negative\n');

% Matriz de valores F1_Score
F1_ScoresMat = [];

% Calculamos la matriz de confusion para cada clase
for i=1:10
    [precision, recall] = matrizConfusion(p,ytest,i);
    F1_Score = 2 * ((precision * recall) / (precision + recall));
    F1_ScoresMat = [ F1_ScoresMat; F1_Score ];
end

% Muestreo del grafico final
figure;
title('Comparativa de clasificadores');
xlabel('Clasificadores');
ylabel('F1_Score');
bar(F1_ScoresMat)
legend ('F1_Score','Location','NorthWest')

% Inventa una solucion y muestra las confusiones
verConfusiones(Xtest, ytest, p);

matriz = [ytest p];

media = mean(F1_ScoresMat);
fprintf('La media de los F1_SCORES: %f\n', media);
