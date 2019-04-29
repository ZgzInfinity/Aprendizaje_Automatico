function [ mejor_lambda ] = validacionCruzada( X, y, nc, NaiveBayes)

 % Se inicializan las variables.
  mejor_lambda = 0;
  mejor_error = inf;

  errores_T = [];
  errores_V = [];

  lambdas = logspace(-10, 0);

  % Se realiza un bucle sobre los valores de lambda.
  for lambda = lambdas    
      
      % Se resetean los valores de los errores.
      err_T = 0;
      err_V = 0;
      
      % Se separa X e Y para obtener datos de entrenamiento y de validacion.
      [ Xcv, ycv, Xtr, ytr ] = particion( 1, 5, X, y );
          
      % Se obtiene el theta que minimiza la funcion de CosteLogReg.
      modelo = entrenarGaussianas(Xtr, ytr, nc, NaiveBayes, lambda);
      
      % Se obtiene la prediccion para la theta obtenida enteriormente con los
      % datos de entrenamiento.
      h = clasificacionBayesiana(modelo, Xtr);   
      
      % Se calculan los errores con los datos de entrenamiento.
      err_T = err_T + ((1 - (mean(double(h == ytr))))*100);;    

      % Se obtiene la prediccion para la theta obtenida enteriormente con los
      % datos de test.
      h = clasificacionBayesiana(modelo, Xcv);
      
      % Se calculan los errores con los datos de test.
      err_V = err_V + ((1 - (mean(double(h == ycv))))*100);
      
      fprintf('Error de entrenamiento %f\n', err_T);
      fprintf('Error de validacion %f\n\n', err_V);
      
      % Se guardan los errores en las matrices de errores correspondientes.
      errores_T = [errores_T err_T];
      errores_V = [errores_V err_V];

      % Si el error con los datos de test son menores que el menor encontrado
      % hasta el momento, se guarda el lambda y error actual como la mejor opcion.
      if (err_V < mejor_error )
          mejor_lambda = lambda;
          mejor_error = err_V;
      end
      
  end

  % Se dibuja la progresion de los valores de lambda.
  figure;
  grid on; hold on;
  title(sprintf('Rojo: errores de entrenamiento; Azul: errores de validacion'));
  xlabel('Valor de lambda');
  ylabel('Errores');
  semilogx(lambdas,errores_T,'r','LineWidth',3);
  semilogx(lambdas,errores_V,'-b','LineWidth',3);
  legend ('Error Entrenamiento', 'Error Validacion','Location','NorthWest');
end
