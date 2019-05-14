function [mu, c, Jotas] = kmeans(D, mu0)
  % D(m,n), m datapoints, n dimensions
  % mu0(K,n) K initial centroids
  %
  % mu(K,n) final centroids
  % c(m) assignment of each datapoint to a class
  
  % Vector de costes del algoritmo
  Jotas = [];
  
  % Vector de iteraciones del algoritmo
  Iteraciones = [];
  
  % Obtencion del numero de muestras y dimensiones
  % m es el numero de muestras
  % n es el numero de dimensiones
  [m n] = size(D);
  
  % Obtencion del valor de los K centroides
  K = size(mu0, 1);
 
  % FASE DE ASIGNACION DE MUESTRAS AL CENTROIDE MAS CERCANO 
  Z = updateClusters(D, mu0);
  
  % Iniciar una matriz de ceros
  Zanteior = zeros(size(Z));
  
  % Actualizacion de los centroides
  mu = updateCentroids(D, Z, size(mu0, 1));
  
  i = 1;
  % Comprobar que el algoritmo ha convergido
  while (isequal(Z, Zanteior) == 0)
    % Muestra por pantalla el numero de iteracion del algoritmo
    
    % Guardado de la
    Zanteior = Z;
   
    % Actualiazacion de los clusteres
    Z = updateClusters(D, mu);
    
    % Actualizar los datos 
    mu = updateCentroids(D, Z, K);
    
    % Sacar las filas de iguales a Z 
    muk = mu(Z,:);

    % Calculo de distancia euclidea de cada punto a la media del centroide correspondiente
    J_i = mean(sqrt((D(:,1)- muk(:,1)).^2 + (D(:,2)- muk(:,2)).^2 + (D(:,3)- muk(:,3)).^2));
    
    % Incorporacion de la iteracion
    Iteraciones = [Iteraciones i];
    
    % Incorporacion del coste actual
    Jotas = [Jotas J_i];
    
    % Aumento de la iteracion del algoritmo
    i = i + 1;
    i
  end
 
  fprintf('Fin del algoritmo de K-medias\n');
  
  c = Z;
  
  figure;
  grid on; hold on;
  plot(Iteraciones, Jotas, '-r', 'LineWidth', 3);
  title('Evolucion coste con respecto iteraciones');
  xlabel('Iteraciones','FontSize',12);                 
  ylabel('Valor de J','FontSize',12); 
  legend('Coste');

 end
