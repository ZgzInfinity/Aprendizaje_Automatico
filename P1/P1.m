
% ---------- PRACTICA 1---------
% -- RUBEN RODRIGUEZ ESTEBAN ---
% ------------ 737215 ----------

close all;
%% Cargar los datos entrenamiento
datos1 = load('PisosTrain.txt');
y1 = datos1(:,3);  % Precio en Euros
x1 = datos1(:,1); % m^2
x2 = datos1(:,2); % Habitaciones
N1 = length(y1);

%% Cargar los datos de test
datos2 = load('PisosTest.txt');
y2 = datos2(:,3);  % Precio en Euros
x3 = datos2(:,1); % m^2
x4 = datos2(:,2); % Habitaciones
N2 = length(y2);

%APARTADO 2

function ecNormalMono(V, S, D, t)
  %% Dibujo de un Ajuste Monovariable test
  figure;
  plot(V, S, 'bx');
  title('Precio de los Pisos')
  ylabel('Euros'); xlabel('Superficie (m^2)');
  grid on; hold on; 

  X = [ones(D,1) V];
  th = X \ S;  % Pongo un valor cualquiera de pesos
  Xextr = [1 min(V)  % Predicción para los valores extremos
         1 max(V)];
  yextr = Xextr * th;  
  plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción
  if (t == 0) 
    legend('Datos Entrenamiento', 'Ecuacion normal')
  else
    legend('Datos Test', 'Ecuacion normal')
  end
  
  'Prediccion dd 100 m2'
  x100 = sum (th .* [1 100]')
  
end

% APARTADO 3

function ecNormalBiDos(V1, V2, S, D)
  %% APARTADO 3
  %% Entrenamiento
  X = [ones(D,1) V1 V2];
  th = X \ S;  % Pongo un valor cualquiera de pesos
  yest = X * th;

  % Dibujar los puntos de entrenamiento y su valor estimado 
  figure;  
  plot3(V1, V2, S, '.r', 'markersize', 20);
  axis vis3d; hold on;
  plot3([V1 V1]' , [V2 V2]' , [S yest]', '-b');

  % Generar una retícula de np x np puntos para dibujar la superficie
  np = 20;
  ejex1 = linspace(min(V1), max(V1), np)';
  ejex2 = linspace(min(V2), max(V2), np)';
  [x1g,x2g] = meshgrid(ejex1, ejex2);
  x1g = x1g(:); %Los pasa a vectores verticales
  x2g = x2g(:);

  % Calcula la salida estimada para cada punto de la retícula
  Xg = [ones(size(x1g)), x1g, x2g];
  yg = Xg * th;

  % Dibujar la superficie estimada
  surf(ejex1, ejex2, reshape(yg,np,np)); grid on;
  title('Precio de los Pisos')
  zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');
  
  % estimacion de las predicciones
  % baja conforme aumentan las habitaciones
  
  'Prediccion 100 m2 y 2 dormitorios'
  x1002 = sum (th .* [1 100 2]')

  'Prediccion 100 m2 y 3 dormitorios'
  x1003 = sum (th .* [1 100 3]')

  'Prediccion 100 m2 y 4 dormitorios'
  x1004 = sum (th .* [1 100 4]')

  'Prediccion 100 m2 y 5 dormitorios'
  x1005 = sum (th .* [1 100 5]')
end

% APARTADO 4

% Coste de Huber
function [J, grad] = CosteHuber(theta,X,y, d, r) 
  good = abs(r) <= d;
  J = (1/2)*sum(r(good).^2) + d*sum(abs(r(~good))) - (1/2)*sum(~good)*d^2;
  if nargout > 1
    grad = X(good,:)'*r(good) + d*X(~good,:)'*sign(r(~good));
  end
 end

% Coste cuadratico
function [J , grad] = CosteL2(theta,X,y,r) 
  % Calcula el coste cuadrático
  J = (1/2)*sum(r.^2);
  if nargout > 1 
    grad = X'*r; 
  end
end

% funcion iterativa del calculo de descenso de gradiente
function [th, i] = iterar (X, th, y, t, C, I)
    alfa = 0.001;
    fi = 50000;
    r = X * th - y;
    % determinar que coste usar
    if (t == 0)  
      % cuadratico
      [nC, g] = CosteL2(th, X, y, r);
    else
      % Huber
      [nC, g] = CosteHuber(th, X, y, fi, r); 
    end
    % Incorporar nuevo coste
    C = [C nC];    
    % Incorporar iteracion  
    i = 1;
    I = [I i];
    do 
      i = i + 1;
      th = th - alfa * g;
      r = X * th - y;
      % determinar que coste usar
      if (t == 0)  
        % cuadratico
        [nC, g] = CosteL2(th, X, y, r);
      else
        % Huber
        [nC, g] = CosteHuber(th, X, y, fi, r); 
      end
      % Incorporar nuevo coste
      C = [C nC];    
      % Incorporar iteracion  
      I = [I i];
    until ((abs(C(i) - C(i-1))) < 0.001)
    
    %Calculo de las media
    medR = mean(abs(r))
    
    % regresion efectuada
    figure;
    plot(I, C, '-b','LineWidth',3);
    if (t == 0)
      title('Descenso de gradiente')
      ylabel('J(theta'); xlabel('Iteracion');
    else
      title('Coste de Huber')
      ylabel('J(theta'); xlabel('Iteracion');
    end
end


function [th] = descensoGradiente(X, th, y, t) 
  % falta hallar th inicial
  % Ejemplo funcion de regresión
  %Normalizar los atributos 
  N = size(X,1); 
  mu = mean(X(:,2:end)); 
  sig = std(X(:,2:end)); 
  X(:,2:end) = (X(:,2:end) - repmat(mu,N,1))./ repmat(sig,N,1);
  % Vector de costes e iteraciones vacios
  C = [];
  I = [];
  % Decision del valor de tasa de aprendizaje alfa
  %Resolver la Regresión
  [th,i] = iterar( X, th', y, t, C, I);
    
  %Des-Normalizarlos pesos 
  th(2:end) = th(2:end)./sig'; 
  th(1)= th(1)-(mu*th(2:end));
  
  %th
  %i
end 


 % TRAZADO DE RESULTADOS
 
 %APARTADO 2
 
 % regresion monovariable con datos de entrenamiento
 % ecuacion normal
 ecNormalMono(x1, y1, N1, 0)
 
 % regresion monovariable con datos de test
 % ecuacion normal
 ecNormalMono(x3, y2, N2, 1)
 
 % APARTADO 3
 
 % regresion multivariable con datos de entrenamiento
 % ecuacion normal
 ecNormalBiDos(x1, x2, y1, N1)
 
 % regresion multivariable con datos de entrenamiento
 % ecuacion normal
 ecNormalBiDos(x3, x4, y2, N2)
 
 % APARTADO 4
 
 % regresion monovariable con datos de entrenamiento
 % descenso de gradiente
 M = [ones(N1,1) x1];
 descensoGradiente(M, [5000 1000], y1, 0)
 
 %th =
 %-3.4313e+004
 %2.6082e+003
 %i =  58
 %medR =   6.9535e+004
 
 % regresion monovariable con datos de test
 % descenso de gradiente
 M = [ones(N2,1) x3];
 descensoGradiente(M, [5000 1000], y2,0)
 
 %th =
 %1.4936e+004
 %2.0086e+003
 %i =  525
 %medR =   6.0287e+004
 
 %APARTADO 5
 
 % regresion multivariable con datos de entrenamiento
 % descenso de gradiente
 M = [ones(N1,1) x1 x2];
 descensoGradiente(M,[5000 1000 50000], y1, 0)
 
 %th =
 %-1.2133e+004
 %3.0287e+003
 %-1.8853e+004
 %i =  233
 %medR =   6.9089e+004
 
 % regresion multivariable con datos de test
 % descenso de gradiente
 M = [ones(N2,1) x3 x4];
 descensoGradiente(M, [5000 1000 50000], y2, 0)
 
 
 %th =
 %1.1571e+005
 %2.6093e+003
 %-4.5301e+004
 %i =  963
 %medR =   4.4044e+004
 
 
 % APARTADO 6 OPCIONAL
 
 % regresion multivariable con datos de entrenamiento
 % coste Huber
 M = [ones(N1,1) x1 x2];
 descensoGradiente(M,[5000 1000 50000], y1, 1)
 
 %th =
 %-1.2133e+004
 %3.0287e+003
 %-1.8853e+004
 %i =  413
 %medR =   6.7718e+004
 
 % regresion multivariable con datos de test
 % coste Huber
 M = [ones(N2,1) x3 x4];
 descensoGradiente(M, [5000 1000 50000], y2, 1)
 
 %th =
 %1.1571e+005
 %2.6093e+003
 %-4.5301e+004
 %i =  1628
 %medR =   4.2394e+004
 
