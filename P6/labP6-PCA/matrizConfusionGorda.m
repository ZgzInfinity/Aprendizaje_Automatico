function matrizConfusionGorda(h, y, n1, n2)
  % Matriz global de confusion
  matrizGorda = [];

  for i = [ n1 , n2 ]
    columna = [];
    for j = [ n1 , n2 ]
      columna = [ columna; (sum(double((h == i)&(y == j)))) ];
    end
    matrizGorda = [ matrizGorda columna ];
  end
  
  fprintf('Matriz de confusion global con los clasificadores\n');
  matrizGorda
end

