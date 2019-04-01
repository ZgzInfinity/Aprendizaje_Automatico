%%
%% Practica 5 - Ruben Rodriguez Esteban
%%

%
% Ejercicio 3 - Bayes ingenuo
%
fprintf('\nEjercicio 3 - Bayes ingenuo.\n');
fprintf('---------------------------------------------------------------------------------\n');

clear ; close all;
warning off;
addpath(genpath('../minfunc'));

% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));

X = X(p,:);
y = y(p);

fprintf('\nValidacion cruzada para Bayes Ingenuo\n');

mejor_lambda = validacionCruzada( 0, 0.1, 50, X, y, 10, 1);

modelo = entrenarGaussianas(X, y, 10, 1, mejor_lambda);
    
p = clasificacionBayesiana(modelo,Xtest);  

error_test = ((1 - (mean(double(p == ytest))))*100);
fprintf('Error con datos de test = %f\n',error_test);

% Calculamos la matriz de confusion para cada clase
for i=1:10
    matrizConfusion(p,ytest,i);
end

% Inventa una solucion y muestra las confusiones
verConfusiones(Xtest, ytest, p);

fprintf('\nPrograma pausado. Pulsa ENTER para continuar con el ejercicio 4.\n');
pause;

%
% Ejercicio 4 - Covarianzas completas
%

fprintf('\nEjercicio 4 - Covarianzas completas.\n');
fprintf('---------------------------------------------------------------------------------\n');

% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));

X = X(p,:);
y = y(p);

fprintf('\nValidacion cruzada con matrices de covarianzas completas\n');

mejor_lambda = validacionCruzada( 0, 0.1, 50, X, y, 10, 0);

modelo = entrenarGaussianas(X, y, 10, 0, mejor_lambda);
    
p = clasificacionBayesiana(modelo,Xtest);  

error_test = ((1 - (mean(double(p == ytest))))*100);
fprintf('Error con datos de test = %f\n',error_test);

% Calculamos la matriz de confusion para cada clase
for i=1:10
    matrizConfusion(p,ytest,i);
end

% Inventa una solucion y muestra las confusiones
verConfusiones(Xtest, ytest, p);

