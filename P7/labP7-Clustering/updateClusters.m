function Z = updateClusters(D,mu)
  % D(m,n), m datapoints, n dimensions
  % mu(K,n) final centroids
  %
  % Z(m) assignment of each datapoint to a class
  
  % Total de muestras de la matriz D
  K = size(mu, 1);
  
  % Poner la matriz Z de dimension N,1 toda a ceros
  Z = zeros(size(D,1), 1);

  % Para cada componente de la matriz D
  for x_i = 1 : size(D, 1)
    % Obtencion de la componente
    X = D(x_i, :);
    % Distancia al centroide
    mejorDist = Inf;
    
    % Promedio de la media
    for mu_i = 1 : K
      % Seleccion de la i-esima media
      mu = c(mu_i, :);
      
      % Distancia euclidea de la media del centroide mu con la muestra X
      distEuclidea  = sqrt(sum((X - mu) .^ 2));
     
      % Comprobar que la distancia euclidea es mejor 
      if distEuclidea < mejorDist
        % Se actualiza la distancia
        mejorDist = distEuclidea;
        
        % Se actualiza el nuevo centroide
        Z(x_i) = mu_i;
      end
    end
  end
end