function [best_size, best_ErrorV] = kfold (N, K, entrada, salida)
  best_size = 0;
  best_ErrorV = 100000;
  
  errores_T = [];
  errores_V = [];
  
  for size = 1:N
    error_T = 0;
    error_V = 0;
    [Xexp] = expandir (entrada, [size 1 1]);
    [M, mu, sig] = normalizar(Xexp);
    for i = 1:K
      [entCV, salCV, entTrain, salTrain] = particion(i , K, M, salida);
      hypo = entTrain \ salTrain;
      error_T = error_T + RMSE(hypo, entTrain, salTrain);
      error_V = error_V + RMSE(hypo, entCV, salCV);
    end
    error_T = error_T / K;
    error_V = error_V / K;
    errores_T = [errores_T error_T];
    errores_V = [errores_V error_V];
    if (error_V < best_ErrorV)
      best_size = size;
      best_ErrorV = error_V;
    end
    fprintf('El mejor polinomio hallado es de grado: %d\n' , size);
    fprintf('El mejor error obtenido %d\n', best_ErrorV); 
  end
  
  fprintf('\nFin algoritmo k-fold\n');
  figure;
  grid on; hold on;
  title(sprintf('Rojo -> errorT. Azul -> errorV'));
  plot(errores_T,'r')
  plot(errores_V,'r')
  