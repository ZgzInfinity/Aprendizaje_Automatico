%% Rubén Rodríguez Esteban
%% NIP - 737215
%% Fecha 7-3-2019

function [ mejor_lambda, mejor_error ] = regu( n, k , Xdatos, ydatos)

mejor_lambda = 1;
mejor_error = 100000;

% vectores de errores de entrenamiento, validacion y lambdas
errores_T = [];
errores_V = [];
lambdas = [];

% Valores iniiales
lambda = 0;
paso = 0.0000001;
lambda_max = 0.00002;

while (lambda<=lambda_max)
    
    % Definimos los parametros para la iteracion de cada lambda.
    lambda = lambda + paso;
    lambdas = [lambdas lambda];
    error_T = 0;
    error_V = 0;
    
    [Xexp] = expandir (Xdatos, [10 5 5]);
    % Normalizar los datos
    [ Xn, mu, sig ] = normalizar( Xexp );
    for i = 1:k
        % Particion de los datos en muestras de entrenamiento y validacion
        [ Xcv, ycv, Xtr, ytr ] = particion( i, k, Xn, ydatos );
        [nrows,ncols] = size(Xtr);
        % Calculo de la prediccion teniendo en cuenta el factor de regulacion
        h = Xtr'*Xtr + lambda*diag([0 ones(1,ncols-1)]);
        theta = h \ (Xtr'*ytr);
        % Calculo de los errores
        error_T = error_T + RMSE (theta, Xtr, ytr);
        error_V = error_V + RMSE (theta, Xcv, ycv);
    end
    
    % Calculo de medias
    error_T = error_T / k;
    error_V = error_V / k;
    
    % Incorporacion de errores a los vectores
    errores_T = [errores_T error_T];
    errores_V = [errores_V error_V];
    
    % Si el error es mas pequeño
    % actualizar el valor de lambda porque mejora
    if (error_V < mejor_error )
        mejor_lambda = lambda;
        mejor_error = error_V;
    end
end

% Muestreo de los resultados
fprintf('\nFin algoritmo regularizacion\n');
fprintf('\nActualizo mejor error y lambda\n');
fprintf('Nuevo mejor lambda: %d\n',lambda);
fprintf('Nuevo mejor error: %d\n',error_V);


% Pintamos la grafica con los errores de entrenamiento y validacion.
figure;
grid on; hold on;
plot(lambdas,errores_T,'-r', 'LineWidth', 3)
plot(lambdas,errores_V,'-b', 'LineWidth', 3)
title('Curva de aprendizaje');
xlabel('Error de entrenamiento','FontSize',12);                 
ylabel('Valor de Lambda','FontSize',12); 
legend('Lambda Train','Lamba Validaion')