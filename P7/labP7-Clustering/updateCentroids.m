function munew = updateCentroids(D, c)
  % D((m,n), m datapoints, n dimensions
  % c(m) assignment of each datapoint to a class
  %
  % munew(K,n) new centroids
  
  [m n] = size(D);
  munew = [];

for i = 1 : K
    idx = find( c== i);
    a = D(idx,1 : n);
    munew = [munew ; mean(a)];
end
end
