function Z = updateClusters(D, mu)
  % D(m,n), m datapoints, n dimensions
  % mu(K,n) final centroids
  %
  % Z(m) assignment of each datapoint to a class
  
  % Total de muestras de la matriz D
  [m n] = size(D);
  K = size(mu, 1)
  A = reshape(repmat(D(:)', K, []), [], 3);
  Km = [1:K]';
  muk = repmat(mu, m, 1);
  
  % Calculo de distancia euclidea de cada punto al centroide
  calculo = sqrt((A(:,1)- muk(:,1)).^2 + (A(:,2)- muk(:,2)).^2 + (A(:,3)- muk(:,3)).^2);
  
  calc2 = reshape(calculo,[K m]);
  
  [~, Z] = min(calc2, [], 1);
end