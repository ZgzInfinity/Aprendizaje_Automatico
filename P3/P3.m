%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EJERCICIO 2 - REGRESION LOGISTICA BASICA %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nEJERCICIO 2 - REGRESION LOGISTICA BASICA\n');
fprintf('----------------------------------------\n');

% Cargamos los datos del fichero exam_data
datos = load('exam_data.txt');

% Separamos el 20% de los datos
y = datos(:, 3);
N = length(y);
X = datos(:, [1, 2]); 
[ Xcv, ycv, Xtr, ytr ] = particion ( 1 , 5 , X, y);

% Resolvemos la regresion logistica con fminunc
[m, n] = size(Xtr);  
Xtr = [ones(m, 1) Xtr];
theta_inicial = zeros(n + 1, 1);

% Definimos las opciones del coste logistico
options = [];
options.display = 'none';

theta = minFunc(@CosteLogistico,theta_inicial,options,Xtr,ytr);

% Mostramos los valores de theta obtenidos
fprintf('\nValores de theta obtenidos: \n');  
fprintf(' %f \n', theta); 

% Calculamos los porcentajes de errores
fprintf('\n');
fprintf('-----------------------\n'); 
h = prediccion(theta, Xtr);
fprintf('\nPorcentaje de errores con datos de entrenamiento: %f\n', ((1 - (mean(double(h == ytr))))*100));

[m, n] = size(Xcv);  
Xcv = [ones(m, 1) Xcv];
h = prediccion(theta, Xcv);
fprintf('Porcentaje de errores con datos de test: %f\n', ((1 - (mean(double(h == ycv))))*100));

% Dibujamos graficamente las muestras y la recta de prediccion
[m, n] = size(X); 
X = [ones(m, 1) X];
plotDecisionBoundary(theta, X, y);  
hold on;  
xlabel('Exam 1 score')  
ylabel('Exam 2 score')  
legend('Admitted', 'Not admitted')  
hold off;

% Calculamos iterativamente la evolucion de la prediccion sabiendo que la
% calificacion del primer examen es 45.
predicciones = [];
cuenta_i = [];

for i = 0:100
    xp = [1 45 i];
    h = 1./(1+exp(-(xp*theta)));
    predicciones = [predicciones h];
    cuenta_i = [cuenta_i i];
end

% Dibujamos la grafica con los puntos obtenidos
figure;
grid on; hold on;
title(sprintf('Predicciones de aprobado con primer examen = 45 en funcion del segundo examen'));
plot(cuenta_i,predicciones,'r')

fprintf('\nPrograma pausado. Pulsa ENTER para continuar con el ejercicio 3.\n');
pause;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EJERCICIO 3 - REGULARIZACION %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n\nEJERCICIO 3 - REGULARIZACION\n');
fprintf('----------------------------\n');

% Cargamos los datos del fichero mchip_data

data = load('mchip_data.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
N = length(y);
p = randperm(N); %reordena aleatoriamente los datos
X = X(p,:);
y = y(p);


% Separamos el 20% de los datos
[ Xcv, ycv, Xtr, ytr ] = particion ( 1 , 5 , X, y);

% Aplicamos validacion cruzada para obtener el mejor valor de lambda
fprintf('\nElegimos el valor de lambda mediante validacion cruzada (Min=-0.01 - Max=0.01)');
warning off;
mejor_lambda = kfoldRegularizacion(-0.01,0.01,0.001,10,Xtr,ytr);

% Entrenamos de nuevo con el valor obtenido de lambda y con lambda=0
Xtr = mapFeature(Xtr(:,1), Xtr(:,2));
theta_inicial = zeros(size(Xtr, 2), 1);

% Resolvemos la regresion mediante 'minFunc'
options = [];
options.display = 'none';
lambda = -0.1;

theta_mejor_lambda = minFunc(@CosteLogReg,theta_inicial,options,Xtr,ytr,mejor_lambda);
theta_lambda_0 = minFunc(@CosteLogReg,theta_inicial,options,Xtr,ytr,0);

%% Dibujar la solicion para lambda=0
X = mapFeature(X(:,1), X(:,2));
plotDecisionBoundary(theta_lambda_0, X, y);
title(sprintf('lambda = %g', 0))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Lambda = 0')

%% Dibujar la solicion para lambda elegida a traves de validacion cruzada
plotDecisionBoundary(theta_mejor_lambda, X, y);
title(sprintf('lambda = %g', mejor_lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Lambda = mejor_lambda')


% Calculamos los errores

% Lambda = 0
% Error con datos de entrenamiento
fprintf('\n-----------------------------------------------------\n');
fprintf('\nErrores con lambda=0');
h = prediccion(theta_lambda_0, Xtr);
error_tr = ((1 - (mean(double(h == ytr))))*100);
fprintf('\nTasa de errores con datos de entrenamiento: %f\n', error_tr);

% Error con datos de test
[m, n] = size(Xcv);  
Xcv = mapFeature(Xcv(:,1), Xcv(:,2));
h = prediccion(theta_lambda_0, Xcv);
error_cv = ((1 - (mean(double(h == ycv))))*100);
fprintf('Tasa de errores con datos de test: %f\n', error_cv);


% Lambda = mejor_lambda
% Error con datos de entrenamiento
fprintf('\nErrores con lambda=%f',mejor_lambda);
h = prediccion(theta_mejor_lambda, Xtr);
error_tr = ((1 - (mean(double(h == ytr))))*100);
fprintf('\nTasa de errores con datos de entrenamiento: %f\n', error_tr);

% Error con datos de test
[m, n] = size(Xcv);  
h = prediccion(theta_mejor_lambda, Xcv);
error_cv = ((1 - (mean(double(h == ycv))))*100);
fprintf('Tasa de errores con datos de test: %f\n', error_cv);

fprintf('\nPrograma pausado. Pulsa ENTER para continuar con el ejercicio 4.\n');
pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EJERCICIO 4 - PRECISION/RECALL %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n\nEJERCICIO 4 - PRECISION/RECALL\n');
fprintf('------------------------------\n');

% Calculamos los datos necesarios para la matriz. La prediccion sobre los
% datos de test ya ha sido realizada en la ultima parte del ejercicio 3 (h).

%TN - True negative
tn = (sum(double((h==0)&(ycv==0))));
%FN - False negative
fn = (sum(double((h==0)&(ycv==1))));
%FP - False positive
fp = (sum(double((h==1)&(ycv==0))));
%TP - True positive
tp = (sum(double((h==1)&(ycv==1))));


matriz_confusion = [tp fp; fn tn]

precision = tp / (tp + fp);
recall = tp / (tp + fn);

fprintf('Precision = %f\n', precision);
fprintf('Recall = %f\n', recall);