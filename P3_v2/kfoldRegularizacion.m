function [ mejor_lambda, mejor_error ] = kfoldRegularizacion ( k, X, Y )
  
  % Se inicializan las variables.
  mejor_lambda=0;
  mejor_error=inf;
  errores_T = [];
  errores_V = [];
  
  lambdas = logspace(-10, 0);
  
  % Se realiza un bucle sobre los valores de lambda a lambda_maxima
 for lambda = lambdas
    
    % Se resetean los valores de los errores.
    err_T = 0;
    err_V = 0;
    
    % Se realiza un bucle sobre k.
    for i = 1:k
      
      % Se separa X normalizado e Y para obtener datos de entrenamiento y de.
      % validacion
      [ Xcv, ycv, Xtr, ytr ] = particion(i, k, X, Y);
      
      % Se obtiene el theta que minimiza la funcion de CosteLogReg.
      options = [];
      options.display = 'none';
      Xtr = mapFeature(Xtr(:,1), Xtr(:,2));
      theta_ini = zeros(size(Xtr, 2), 1);
      theta = minFunc(@CosteLogReg,theta_ini,options,Xtr,ytr,lambda);
      
      % Se obtiene la prediccion para la theta obtenida enteriormente.
      h = prediccion(theta, Xtr);
      
      % Se calculan los errores
      err_T = err_T + ((1 - (mean(double(h == ytr))))*100);
      
      % Error con datos de validacion
      [m, n] = size(Xcv);  
      Xcv = mapFeature(Xcv(:,1), Xcv(:,2));
      h = prediccion(theta, Xcv);
      err_V = err_V + ((1 - (mean(double(h == ycv))))*100);
    end
    
    % Se calculan los errores medios de las k veces.
    err_T = err_T/k;
    err_V = err_V/k;
    
    % Se almacenan los error para luego dibujarlas.
    errores_T = [errores_T err_T];
    errores_V = [errores_V err_V];
    
    % Si el error con los datos de validacion es menor que el menor error 
    % encontrado, se guarda como el menor error encontrado y se guarda el grado.
    if (err_V < mejor_error)
      mejor_lambda = lambda;
      mejor_error = err_V;
    end
  end
  
  % Una vez encontrados los valores, se muestran la grafica con los errores
  % obtenidos.
  figure;
  grid on; hold on;
  title(sprintf('Rojo: errores entrenamiento | Azul: errores validacion'));
  ylabel('Error'); xlabel('Lambda');
  semilogx(lambdas,errores_T,'r')
  semilogx(lambdas,errores_V)
  
end


