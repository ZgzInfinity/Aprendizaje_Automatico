function [mu, c] = kmeans(D, mu0)
  % D(m,n), m datapoints, n dimensions
  % mu0(K,n) K initial centroids
  %
  % mu(K,n) final centroids
  % c(m) assignment of each datapoint to a class
  
  % Obtencion del numero de muestras y dimensiones
  % m es el numero de muestras
  % n es el numero de dimensiones
  [m n] = size(D);
  
  % Obtencion del valor de los K centroides
  K = size(mu0, 1);
  
  % FASE DE ASIGNACIO DE MUESTRAS AL CENTROIDE MAS CERCANO 
  Z = updateClusters(D, mu0);
  
  % Comprobar que el algoritmo ha convergido
  while (c_iActual ~= c_iAnterior)
    % Muestra por pantalla el numero de iteracion del algoritmo
    fprintf('Iteracion del algoritmo K-medias : %d \n', i);
    
    % Guardado de la
    c_iAnterior = c_iActual;
    
    % Actualiazacion de los clusteres
    Z = updateClusters(D, c);
    
    % Actualizar los centroides
    c = updateCentroids(D, Z, size(c0, 1));   
    
    % Aumento de la iteracion del algoritmo
    i = i + 1;
  end
  
  fprintf('Fin del algoritmo de K-medias\n');

 end
