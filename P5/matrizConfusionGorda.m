function matrizConfusionGorda(h, y, nClases)
  % Matriz global de confusion
  matrizGorda = [];

  for i = 1:nClases
    columna = [];
    for j = 1:nClases
      columna = [ columna; (sum(double((h == i)&(y == j)))) ];
    end
    matrizGorda = [ matrizGorda columna ];
  end
  
  fprintf('Matriz de confusion global con los clasificadores');
  matrizGorda
end

