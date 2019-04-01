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

mejor_lambda = validacionCruzada(X, y, 10, 1);

modelo = entrenarGaussianas(X, y, 10, 1, mejor_lambda);
    
p = clasificacionBayesiana(modelo,Xtest);  

error_test = ((1 - (mean(double(p == ytest))))*100);
fprintf('Error con datos de test = %f\n',error_test);

% Matriz de valores F1_Score
F1_ScoresMat = [];
PrecisionMat = [];
RecallMat = [];

% Calculamos la matriz de confusion para cada clase
for i=1:10
    [precision, recall] = matrizConfusion(p,ytest,i);
    F1_Score = 2 * ((precision * recall) / (precision + recall));
    F1_ScoresMat = [ F1_ScoresMat; F1_Score ];
    PrecisionMat = [PrecisionMat; precision ];
    RecallMat = [RecallMat; recall ];
end

% Muestreo del grafico final
figure;
title('Comparativa de clasificadores en Bayes ingenuo');
xlabel('Clasificadores');
ylabel('F1_Score');
bar(F1_ScoresMat)
legend ('F1_Score','Location','NorthWest')


% Inventa una solucion y muestra las confusiones
verConfusiones(Xtest, ytest, p);

% obtencion de la matriz de confusion global
matrizConfusionGorda(p, ytest, 10);

% Resultados analiticos
mediaPrecision = mean(PrecisionMat);
mediaRecall = mean(RecallMat);
mediaScore = mean(F1_ScoresMat);
fprintf('La media de la precision: %f\n',mediaPrecision);
fprintf('La media del recall: %f\n',mediaRecall);
fprintf('La media de los F1_SCORES: %f\n', mediaScore);


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

mejor_lambda = validacionCruzada(X, y, 10, 0);

modelo = entrenarGaussianas(X, y, 10, 0, mejor_lambda);
    
p = clasificacionBayesiana(modelo,Xtest);  

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
PrecisionMat = [];
RecallMat = [];

% Calculamos la matriz de confusion para cada clase
for i=1:10
    [precision, recall] = matrizConfusion(p,ytest,i);
    F1_Score = 2 * ((precision * recall) / (precision + recall));
    F1_ScoresMat = [ F1_ScoresMat; F1_Score ];
    PrecisionMat = [PrecisionMat; precision ];
    RecallMat = [RecallMat; recall ];
end


% Muestreo del grafico final
figure;
title('Comparativa de clasificadores en Covarianzas completas');
xlabel('Clasificadores');
ylabel('F1_Score');
bar(F1_ScoresMat)
legend ('F1_Score','Location','NorthWest')


% Inventa una solucion y muestra las confusiones
verConfusiones(Xtest, ytest, p);


% obtencion de la matriz de confusion global
matrizConfusionGorda(p, ytest, 10);

% Resultados analiticos
mediaPrecision = mean(PrecisionMat);
mediaRecall = mean(RecallMat);
mediaScore = mean(F1_ScoresMat);
fprintf('La media de la precision: %f\n',mediaPrecision);
fprintf('La media del recall: %f\n',mediaRecall);
fprintf('La media de los F1_SCORES: %f\n', mediaScore);
