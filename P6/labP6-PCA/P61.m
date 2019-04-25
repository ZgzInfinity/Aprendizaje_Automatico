%% Practica 6.1: PCA 

clear all
close all

% Leer los datos originales en la variable X
load P61

% Graficar los datos originales
figure(1);
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

printf('La matriz de atributos estandarizados es la siguiente ');
Xest

% Graficar los datos centrados
figure(2);
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
vDiag = diag(V)'

printf('La matriz de los vectores propios ordenada de mayor a menor\n ');
printf('segun su valor propio es la siguiente\n');
[~, I] = sort(vDiag, 'descend');
Uord = U(:,I')

% Graficar en color rojo cada vector propio * 3 veces la raiz de su 
% correspondiente valor propio
printf('La matrz resultante de multiplicar la raiz cuadradada de cada valor es la siguiente\n');
vDiagMod = 3 * sqrt(vDiag)

printf('La matriz resultantes de cada vector propio * 3 veces la raiz de su valor propio es la siguiente\n');
res = Uord * diag(vDiagMod)

% Graficar la variabilidad que se mantiene si utilizas los tres primeros
% vectores propios, los dos primeros, o solo el primer vector propio

figure(3);
axis equal;
grid on;
hold on;
plot(res(:,1),'-r');
plot(Xest(:,1),'b.');
xlabel('Xest');
ylabel('Yest');
zlabel('Zest');


%figure(4);
%axis equal;
%grid on;
%hold on;
%plot2(res(:,1),res(:,2),'-r');
%plot2(Xest(:,1),Xest(:,2),'b.');
%xlabel('Xest');
%ylabel('Yest');
%zlabel('Zest');


figure(5);
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

kN = sum(diag(vDiagMod));
N = size(vDiagMod);

encontrado = 0;
k = 1;

while (k < N && encontrado != 1)
  for i = 1:k
      sumaK = sumaK + vDiagMod(i , i);
  end 
  if (sumaK / kN >= 0.90)
    encontrado = 1;
  end 
end


% Graficar aparte los datos z proyectados según el resultado anterior

% Graficar en verde los datos reproyectados \hat{x} en la figura original
