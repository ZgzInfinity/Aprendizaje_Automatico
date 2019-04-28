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
plot3(X(:,1),X(:,2),X(:,3),'b.');
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
xlabel('Xest');
ylabel('Yest');
zlabel('Zest');

% Calcular la matrix de covarianza muestral de los datos centrados
mCov = cov(Xest);

printf('La matriz de covarianza de los atributos estandarizados es la siguiente\n ');
mCov

% Aplicar PCA para obtener los vectores propios y valores propios
[U, V] = eig(mCov);

printf('La matriz de vectores propios U es la siguiente\n');
U

printf('La matriz de valores propios es la siguiente\n');
V

% Ordenar los vectores y valores proprios de mayor a menor valor propio
printf('La matriz de los valores propios ordenada de mayor a menor es la siguiente\n ');
vDiagAscend = diag(sort(diag(V)','descend'))
vDiag = sort(diag(V)','descend');
vDiagRef = diag(V)';

printf('La matriz de los vectores propios ordenada de mayor a menor\n ');
printf('segun su valor propio es la siguiente\n');
[~, I] = sort(vDiagRef, 'descend');
Uord = U(:,I')

% Graficar en color rojo cada vector propio * 3 veces la raiz de su 
% correspondiente valor propio
printf('La matrz resultante de multiplicar la raiz cuadradada de cada valor es la siguiente\n');
vDiagMod = 3 * sqrt(sort(vDiagRef,'descend'))

printf('La matriz resultantes de cada vector propio * 3 veces la raiz de su valor propio es la siguiente\n');
res = Uord * diag(vDiagMod)

% Graficar la variabilidad que se mantiene si utilizas los tres primeros
% vectores propios, los dos primeros, o solo el primer vector propio

figure();
title('Variabilidad usando un vector propio');
axis equal;
grid on;
hold on;
plot(res(:,1),'-r');
plot(Xest(:,1),'b.');
xlabel('Xest');
ylabel('Yest');
zlabel('Zest');


%figure();
%axis equal;
%title('Variabilidad usando dos vectores propios');
%grid on;
%hold on;
%plot2(res(:,1),res(:,2),'-r');
%plot2(Xest(:,1),Xest(:,2),'b.');
%xlabel('Xest');
%ylabel('Yest');
%zlabel('Zest');


figure();
title('Variabilidad usando 3 vectores propios');
axis equal;
grid on;
hold on;
plot3(res(:,1),res(:,2),res(:,3),'b.');
plot3(Xest(:,1),Xest(:,2),Xest(:,3),'b.');
xlabel('Xest');
ylabel('Yest');
zlabel('Zest');

% Aplicar PCA para reducir las dimensiones de los datos y mantener al menos
% el 90% de la variabilidad

[k, sumas, sumaK] = findValorKvect(vDiag, 0.90);

fprintf('Las sumas realizadas hasta ahora alcanzar la mejor dimension es la siguiente\n');
sumas

fprintf('La suma primera suma que ha sido superior a 0.90 ha sido %f\n', sumaK);

% Graficar aparte los datos z proyectados según el resultado anterior
Ureduce = Uord(1:size(Uord,1),1:k);

fprintf('Para el numero de dimensiones k = %d\n', k);
fprintf('La matriz de U reducida es la siguiente\n');
Ureduce

% Obtencion de la matriz Z 
Z = Ureduce' * Xest';

figure();
plot3(Z(:,1),Z(:,2),Z(:,3),'b.');



% Graficar en verde los datos reproyectados \hat{x} en la figura original
