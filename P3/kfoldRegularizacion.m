function [ mejor_tam ] = kfoldRegularizacion( lambda_min, lambda_max, paso, k , Xdatos, ydatos)

% Definimos los parametros iniciales
fprintf('\nInicio validacion cruzada.');
mejor_tam = lambda_min;
lambda = lambda_min;
mejor_error = 100000;

% Vectores donde almacenaremos los distintos valores de los errores y de
% las lambdas recorridas. Dichos vectores los usaremos para pintar las
% graficas de evolucion de dichos valores.
errores_T = [];
errores_V = [];
lambdas = [];

% Comienzo del algoritmo de validacion cruzada
while (lambda<=lambda_max)
    % Definimos los parametros para la iteracion de cada lambda.
    lambda = lambda + paso;
    lambdas = [lambdas lambda];
    error_T = 0;
    error_V = 0;
    
    for i = 1:k
        % Dividimos los datos entre entrenamiento y validacion
        [ Xcv, ycv, Xtr, ytr ] = particion( i, k, Xdatos, ydatos );
        options = [];
        options.display = 'none';
        Xtr = mapFeature(Xtr(:,1), Xtr(:,2));
        theta_inicial = zeros(size(Xtr, 2), 1);
        
        % Resolvemos la regresion mediante 'minFunc'
        theta = minFunc(@CosteLogReg,theta_inicial,options,Xtr,ytr,lambda);
        
        % Calculamos los errores.
        % Error con datos de entrenamiento
        h = prediccion(theta, Xtr);
        error_tr = ((1 - (mean(double(h == ytr))))*100);
        error_T = error_T + error_tr;
        
        % Error con datos de validacion
        [m, n] = size(Xcv);  
        Xcv = mapFeature(Xcv(:,1), Xcv(:,2));
        h = prediccion(theta, Xcv);
        error_cv = ((1 - (mean(double(h == ycv))))*100);
        error_V = error_V + error_cv;
        
    end
     
     % Calculamos la media de los errores
     error_T = error_T / k;
     error_V = error_V / k;
     
     % Informamos por pantalla de los resultados obtenidos para lambda
     fprintf('\nLambda = %f \n', lambda);
     fprintf('Error con lambda %f (Entrenamiento): %f\n', lambda, error_T);
     fprintf('Error con lambda %f (Verificacion): %f\n', lambda, error_V);
     
     % Añadimos los valores de los errores a sus respectivos vectores
     errores_T = [errores_T error_T];
     errores_V = [errores_V error_V];
     
    % Si el valor de lambda mejora el porcentaje de errores con los datos de validacion 
    % actualizamos los valores de mejor lambda y mejor error.
    if (error_V < mejor_error )
        mejor_tam = lambda;
        mejor_error = error_V;
    end
    
end

% Fin del algoritmo. Informamos por pantalla de los resultados obtenidos.
fprintf('\nFin validacion cruzada.');
fprintf('\nMejor lambda : %f', mejor_tam);
fprintf('\nError con datos de validacion : %f\n', mejor_error);

% Pintamos la grafica con los errores de entrenamiento y validacion.
figure;
grid on; hold on;
title(sprintf('Rojo -> Errores entrenamiento. Azul -> Errores validacion'));
plot(log10(lambdas),errores_T,'r')
plot(log10(lambdas),errores_V)