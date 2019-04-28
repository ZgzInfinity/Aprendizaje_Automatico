%% Practica 6.2: PCA 

clear all
close all

% Leer la imagen
I = imread('turing.png');

% Convertirla a blanco y negro
BW = rgb2gray(I);

% Convertir los datos a double
X=im2double(BW);

% graficar la imagen
figure(1);
colormap(gray);
imshow(X);
axis off;

% Aplicar PCA

% Centrar la matriz con respecto de la media
mu = mean(X);
Xest = X - mu;

% Calculo de la covarianza
mCov = cov(Xest);

% Calculo de las matrices U, S, V con svd
[U,S,V] = svd(mCov);

% Calculo del valor de la matriz Z con en las nuevas dimensiones
Z = U' * Xest';

% Calculo del valor de Xhat 
Xgorro = U * Z;
Xhat = Xgorro';

% Graficar las primeras 5 componentes
for k = 1:5,
    figure(2);
    imshow(Xhat);
    colormap(gray);
    axis off;
end

% Graficar la reconstrucción con las primeras 1, 2, 5, 10, 20, y total
% de componentes
for k = [1 2 5 10 20 rank(X)],
    figure(3);
    imshow(Xhat);
    colormap(gray);
    axis off;
end


% Encontrar el valor de k que mantenga al menos el 90% de la variabilidad
[k, sumas, sumaK] = findValorKmat(V, 0.90);

fprintf('Las sumas realizadas hasta ahora alcanzar la mejor dimension es la siguiente\n');
sumas

fprintf('La suma primera suma que ha sido superior a 0.90 ha sido %f\n', sumaK);


% Graficar la reconsrtucción con las primeras k componentes


% Calcular y mostrar el ahorro en espacio
