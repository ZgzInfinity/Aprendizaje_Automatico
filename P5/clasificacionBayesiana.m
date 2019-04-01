
function yhat = clasificacionBayesiana(modelo, X)
% Con los modelos entrenados, predice la clase para cada muestra X

predicciones = [];
for(i=1:10)
    predicciones = [predicciones gaussLog(modelo{i}.mu, modelo{i}.Sigma, X)];
end

yhat=[];
for(i=1:size(X,1))
% Seleccionamos la prediccion que mas se ajusta de cada clase
    [maxval, maxindice] = max(predicciones(i,:));
    yhat = [yhat maxindice];
end

yhat = yhat';


