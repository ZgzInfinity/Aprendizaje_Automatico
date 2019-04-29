function [ precision, recall] = matrizConfusion( p, y, n1 , n2)

  % Vector de cocientes F_score
  F1_ScoresMat = [];

  for clase = [ n1 , n2 ]
      %TN - True negative
      tn = (sum(double((p~=clase)&(y~=clase))));
      %FN - False negative
      fn = (sum(double((p~=clase)&(y==clase))));
      %FP - False positive
      fp = (sum(double((p==clase)&(y~=clase))));
      %TP - True positive
      tp = (sum(double((p==clase)&(y==clase))));
      fprintf('------------------------\n\n');
      fprintf('Para el clasificador %d\n', clase);
      matriz_confusion = [tp fp; fn tn]

      precision = tp / (tp + fp);
      recall = tp / (tp + fn);
      
      % Calculo del F_score para cada clase estudiada
      F1_Score = 2 * ((precision * recall) / (precision + recall));
      F1_ScoresMat = [ F1_ScoresMat; F1_Score ];

      % Muestreo de estadisticas
      
      fprintf('Precision (%d) = %f\n', clase, precision);
      fprintf('Recall (%d) = %f\n', clase, recall);
      fprintf('F_Score (%d) = %f\n', clase, F1_Score);
  end

  figure;
  title('Comparativa de clases');
  xlabel('Clases');
  ylabel('F1_Score');
  bar(F1_ScoresMat)
  legend ('F1_Score','Location','NorthWest')
  
end