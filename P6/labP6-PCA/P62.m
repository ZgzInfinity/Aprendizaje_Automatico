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

% Ordenar la matriz de valores propios de mayor a menor
sDiagAscend = diag(sort(diag(S)','descend'));
sDiag = sort(diag(S)','descend');
sDiagRef = diag(S)';


% La matriz de los vectores propios ordenada de mayor a menor
% segun su valor propio es la siguiente
[~, I] = sort(sDiagRef, 'descend');
Uord = U(:,I');


%Identificador de imagen
idImagen = 1;

% Graficar las primeras 5 componentes
for k = 1:5,
    figure(idImagen);
    
    % Coger la K primeras columnas de la matriz U ordenada
    Ureduce = Uord(1:size(Uord,1),1:k);
    
    % Obtencion de la matriz Z para K dimensiones
    Z = Ureduce' * Xest';
    Zprim = Z';
    
    % Reconstruccion de los datos para k dimensiones a partir de Z
    Xgorro = Ureduce * Z;
    Xhat = Xgorro';
    imshow(Xhat);
    colormap(gray);
    axis off;
    
    %Incremento del id de la imagen
    idImagen = idImagen + 1;
    
    % Parada intermedia de muestreo de imagenes
    pause(0.5);
end

% Graficar la reconstrucción con las primeras 1, 2, 5, 10, 20, y total
% de componentes
for k = [1 2 5 10 20 rank(X)],
    figure(idImagen);
    % Coger la K primeras columnas de la matriz U ordenada
    Ureduce = Uord(1:size(Uord,1),1:k);
    
    % Obtencion de la matriz Z para K dimensiones
    Z = Ureduce' * Xest';
    Zprim = Z';
    
    % Reconstruccion de los datos para k dimensiones a partir de Z
    Xgorro = Ureduce * Z;
    Xhat = Xgorro';
    imshow(Xhat);
    colormap(gray);
    axis off;
    
    %Incremento del id de la imagen
    idImagen = idImagen + 1;
    
    % Parada intermedia de muestreo de imagenes
    pause(0.5);
end


% Encontrar el valor de k que mantenga al menos el 90% de la variabilidad
[k] = findValorKvect(sDiag, 0.90);

fprintf('El valor de k es el siguiente: %d\n', k);
pause;

% Graficar la reconstrucción con las primeras k componentes
for i = 1:3,
    figure(idImagen);
    
    % Coger la I primeras columnas de la matriz U ordenada
    Ureduce = Uord(1:size(Uord,1),1:i);
    
    % Obtencion de la matriz Z para K dimensiones
    Z = Ureduce' * Xest';
    Zprim = Z';
    
    % Reconstruccion de los datos para k dimensiones a partir de Z
    Xgorro = Ureduce * Z;
    Xhat = Xgorro';
    imshow(Xhat);
    colormap(gray);
    axis off;
    
    %Incremento del id de la imagen
    idImagen = idImagen + 1;
    
    % Parada intermedia de muestreo de imagenes
    pause(0.5);
end

% Calcular y mostrar el ahorro en espacio
plot(diag(S));