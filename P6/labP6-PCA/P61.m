%% Practica 6.1: PCA 

clear all
close all

% Leer los datos originales en la variable X
load P61

% Graficar los datos originales
figure();
title('Grafico con los datos originales');
axis equal;
grid on;
hold on;
plot3(X(:,1),X(:,2),X(:,3),'k.');
xlabel('X');
ylabel('Y');
zlabel('Z');

% Estandarizar los datos (solo hace falta centrarlos)
mu = mean(X);
Xest = X - mu;

% Graficar los datos centrados
figure();
title('Grafico con los datos originales centrados a la media');
axis equal;
grid on;
hold on;
plot3(Xest(:,1),Xest(:,2),Xest(:,3),'b.');
xlabel('Eje X ');
ylabel('Eje Y');
zlabel('Eje Z');

% Calcular la matrix de covarianza muestral de los datos centrados
mCov = cov(Xest);

printf('La matriz de covarianza de los atributos estandarizados es la siguiente\n ');
mCov

% Aplicar PCA para obtener los vectores propios y valores propios
[U, V] = eig(mCov);


% Ordenar los vectores y valores proprios de mayor a menor valor propio
vDiagAscend = diag(sort(diag(V)','descend'));
vDiag = sort(diag(V)','descend');
vDiagRef = diag(V)';

% La matriz de los vectores propios ordenada de mayor a menor
% segun su valor propio es la siguiente
[~, I] = sort(vDiagRef, 'descend');
Uord = U(:,I');

% Graficar en color rojo cada vector propio * 3 veces la raiz de su 
% correspondiente valor propio
vDiagMod = 3 * sqrt(sort(vDiagRef,'descend'));
res = Uord * diag(vDiagMod);


% Graficar la variabilidad que se mantiene si utilizas los tres primeros
% vectores propios, los dos primeros, o solo el primer vector propio


% FALTA ESTO


% Aplicar PCA para reducir las dimensiones de los datos y mantener al menos
% el 90% de la variabilidad

[k] = findValorKvect(vDiag, 0.90);

fprintf('El mejor valor hallado de k es %d\n', k);


% Graficar aparte los datos z proyectados según el resultado anterior
Ureduce = Uord(1:size(Uord,1),1:k);


% Obtencion de la matriz Z 
Z = Ureduce' * Xest';
Zprim = Z';


figure();
title('Grafico de los datos Z');
axis equal;
grid on;
hold on;
plot(Zprim(:,1),Zprim(:,2),'r.');
xlabel('Eje X');
ylabel('Eje Y');

% Graficar en verde los datos reproyectados \hat{x} en la figura original
Xgorro = Ureduce * Z;
Xhat = Xgorro';

figure();
title('Reconstruccion de los datos originales');
axis equal;
grid on;
hold on;
plot(Zprim(:,1),Zprim(:,2),'r.');
plot3(Xest(:,1),Xest(:,2),Xest(:,3),'b.');
plot3(Xhat(:,1),Xhat(:,2),Xhat(:,3),'g.');
xlabel('Eje X');
ylabel('Eje Y');
zlabel('Eje Z');
