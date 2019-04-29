
function yhat = clasificacionBayesiana(modelo, X, n1, n2)
  % Con los modelos entrenados, predice la clase para cada muestra X

  predicciones = [];
  
  for i = [n1 , n2]
      predicciones = [predicciones gaussLog(modelo{i}.mu, modelo{i}.Sigma, X)];
  end 
  
  yhat=[];
  valores = [n1 , n2];

  % Bucle con los dos valores
  for (i = 1:size(X,1))
  % Seleccionamos la prediccion que mas se ajusta de cada clase
      [maxval, maxindice] = max(predicciones(i,:));
      if (maxindice == 1)
        yhat = [yhat n1];
      else
        yhat = [yhat n2];
      end
  end

  yhat = yhat';

end
