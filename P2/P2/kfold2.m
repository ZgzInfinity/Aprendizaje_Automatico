%% Rubén Rodríguez Esteban
%% NIP - 737215
%% Fecha 7-3-2019

function [best_size, best_ErrorV] = kfold2 (N, K, entrada, salida, best_Last)
  % Mejor modelo polinomico
  best_size = 0;
  % Mejor error encontrado
  best_ErrorV = 100000;
  
  % vectores de errores de entrenamiento y validacion
  errores_T = [];
  errores_V = [];
  
  % Para cada modelo
  for size = 1:N
    error_T = 0;
    error_V = 0;
    [Xexp] = expandir (entrada, [best_Last size 1]);
    % Normalizacion
    [M, mu, sig] = normalizar(Xexp);
    for i = 1:K
      % Division de los datos en conjunto de entrenamiento y validacion
      [entCV, salCV, entTrain, salTrain] = particion(i , K, M, salida);
      % Calculo de prediccion con ecuacion normal
      hypo = entTrain \ salTrain;
      % Calculo de errores
      error_T = error_T + RMSE(hypo, entTrain, salTrain);
      error_V = error_V + RMSE(hypo, entCV, salCV);
    end
    
    % Obtecion de las medias 
    error_T = error_T / K;
    error_V = error_V / K;
    
    % Incorporacion a los vectores
    errores_T = [errores_T error_T];
    errores_V = [errores_V error_V];
    
    % Si el error de validacion mejora
    % el modelo es mejor que los anteriores
    if (error_V < best_ErrorV)
      best_size = size;
      best_ErrorV = error_V;
    end
  end
  
  fprintf('\nFin algoritmo k-fold\n');
  fprintf('El mejor polinomio hallado es de grado: %d\n' , size);
  fprintf('El mejor error obtenido %d\n', best_ErrorV); 
  
  % muestreo de los resultados
  figure;
  grid on; hold on;
  plot(errores_T,'-r','LineWidth', 3)
  plot(errores_V,'-b','LineWidth', 3)
  title('Kross Validation -- Distancia en Km');
  xlabel('Distancia en km','FontSize',12);                 
  ylabel('Errores','FontSize',12);
  legend('Error Train','Error Validation')
  