
function [ mejor_lambda, mejor_error ] = regu( n, k , Xdatos, ydatos)

mejor_lambda = 1;
mejor_error = 100000;

errores_T = [];
errores_V = [];
lambdas = [];

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
    [ Xn, mu, sig ] = normalizar( Xexp );
    for i = 1:k
        [ Xcv, ycv, Xtr, ytr ] = particion( i, k, Xn, ydatos );
        [nrows,ncols] = size(Xtr);
        h = Xtr'*Xtr + lambda*diag([0 ones(1,ncols-1)]);
        theta = h \ (Xtr'*ytr);
        error_T = error_T + RMSE (theta, Xtr, ytr);
        error_V = error_V + RMSE (theta, Xcv, ycv);
    end
    error_T = error_T / k;
    error_V = error_V / k;
    errores_T = [errores_T error_T];
    errores_V = [errores_V error_V];
    if (error_V < mejor_error )
        mejor_lambda = lambda;
        mejor_error = error_V;
    end
end

fprintf('\nFin algoritmo regularizacion\n');
fprintf('\nActualizo mejor error y lambda\n');
fprintf('Nuevo mejor lambda: %d\n',lambda);
fprintf('Nuevo mejor error: %d\n',error_V);


% Pintamos la grafica con los errores de entrenamiento y validacion.
figure;
grid on; hold on;
title(sprintf('Rojo -> Errores entrenamiento. Azul -> Errores validacion'));
plot(lambdas,errores_T,'-b', 'LineWidth', 3)
plot(lambdas,errores_V,'-g', 'LineWidth', 3)