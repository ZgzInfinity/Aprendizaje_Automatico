function [ precision, recall ] = matrizConfusion( p, y, clase )

%TN - True negative
tn = (sum(double((p~=clase)&(y~=clase))));
%FN - False negative
fn = (sum(double((p~=clase)&(y==clase))));
%FP - False positive
fp = (sum(double((p==clase)&(y~=clase))));
%TP - True positive
tp = (sum(double((p==clase)&(y==clase))));
fprintf('------------------------\n\n');
fprintf('Matriz de confusion de %d', clase);
matriz_confusion = [tp fp; fn tn]

precision = tp / (tp + fp);
recall = tp / (tp + fn);

fprintf('Precision (%d) = %f\n', clase, precision);
fprintf('Recall (%d) = %f\n', clase, recall);


end
