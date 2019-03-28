function [ mejor_tam ] = validacionCruzada( lambda_min, lambda_max, paso, Xdatos, ydatos, numClases)

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
    
    % Dividimos los datos entre entrenamiento y validacion
    [ Xcv, ycv, Xtr, ytr ] = particion( 1, 5, Xdatos, ydatos );
        
    % Obtenemos theta para el clasificador multiclase
    theta = entrenadorMulticlase(Xtr, ytr, numClases, lambda);
    
    % Calculamos los errores.
    % Error con datos de entrenamiento
    
    p = clasificacionMulticlase(theta, Xtr);   
    error_tr = ((1 - (mean(double(p == ytr))))*100);
    error_T = error_T + error_tr;
    
    % Error con datos de validacion
    p = prediccion(theta, Xcv);
    error_cv = ((1 - (mean(double(p == ycv))))*100);
    error_V = error_V + error_cv;
        
     
    % Calculamos la media de los errores
    error_T = error_T;
    error_V = error_V;
     
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
plot(lambdas,errores_T,'-r','LineWidth',3);
plot(lambdas,errores_V,'-b','LineWidth',3);
title('Comparativa de errores');
xlabel('Valor de lambda');
ylabel('Errores');
legend ('Error Entrenamiento', 'Error Validacion','Location','NorthWest');
plot(lambdas,errores_T,'r')
plot(lambdas,errores_V)
