
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

function ecNormalMono(VTrain, VTest, STr, STest, DTrain, DTest)
  %% Dibujo de un Ajuste Monovariable test
  figure;
  plot(VTrain, STr, 'bx');
  title('Precio de los Pisos')
  ylabel('Euros'); xlabel('Superficie (m^2)');
  grid on; hold on; 

  X = [ones(DTrain,1) VTrain];
  th = X \ STr;  % Pongo un valor cualquiera de pesos
  Xextr = [1 min(VTrain)  % Predicción para los valores extremos
         1 max(VTrain)];
  yextr = Xextr * th;  
  plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción
  legend('Datos Entrenamiento', 'Ecuacion normal')

  fprintf('Calculo de la prediccion\n\n')
  th
  
  r = X * th - STr;
  medR = mean(abs(r));

  fprintf('Error medio de los datos de entrenamiento %d \n\n', medR)
  
  X = [ones(DTest,1) VTest];
  r = X * th - STest;
  medR = mean(abs(r));

  fprintf('Error medio de los datos de test %d \n\n', medR)
  
end

% APARTADO 3

function ecNormalBiDos(V1Train, V2Train, V1Test, V2Test, STrain, STest, DTrain, DTest)
  %% APARTADO 3
  %% Entrenamiento
  X = [ones(DTrain,1) V1Train V2Train];
  th = X \ STrain;  % Pongo un valor cualquiera de pesos
  yest = X * th;

  % Dibujar los puntos de entrenamiento y su valor estimado 
  figure;  
  plot3(V1Train, V2Train, STrain, '.r', 'markersize', 20);
  axis vis3d; hold on;
  plot3([V1Train V1Train]' , [V2Train V2Train]' , [STrain yest]', '-b');

  % Generar una retícula de np x np puntos para dibujar la superficie
  np = 20;
  ejex1 = linspace(min(V1Train), max(V1Train), np)';
  ejex2 = linspace(min(V2Train), max(V2Train), np)';
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
  
  fprintf('Calculo de la prediccion\n');
  th
  
  r = X * th - STrain;
  medR = mean(abs(r));
  
  fprintf('Error de entrenamiento %d\n', medR)
  
  x1002 = sum (th .* [1 100 2]');
  fprintf('Prediccion 100 m2 y 2 dormitorios %d\n', x1002) 
  
  x1003 = sum (th .* [1 100 3]');
  fprintf('Prediccion 100 m2 y 3 dormitorios %d\n', x1003) 
  
  x1004 = sum (th .* [1 100 4]');
  fprintf('Prediccion 100 m2 y 4 dormitorios %d\n', x1004) 
  
  x1005 = sum (th .* [1 100 5]');
  fprintf('Prediccion 100 m2 y 5 dormitorios %d\n', x1005) 
  
  X = [ones(DTest,1) V1Test V2Test];
  r = X * th - STest;
  medR = mean(abs(r));
  
  fprintf('Valor del error del test %d\n\n', medR)
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
function [th, i] = iterar (X, th, yTrain, t, C, I, DTrain, DTest)
    alfa = 0.001;
    fi = 50000;
    r = X * th - yTrain;
    % determinar que coste usar
    if (t == 0)  
      % cuadratico
      [nC, g] = CosteL2(th, X, yTrain, r);
    else
      % Huber
      [nC, g] = CosteHuber(th, X, yTrain, fi, r); 
    end
    % Incorporar nuevo coste
    C = [C nC];    
    % Incorporar iteracion  
    i = 1;
    I = [I i];
    do 
      i = i + 1;
      th = th - alfa * g;
      r = X * th - yTrain;
      % determinar que coste usar
      if (t == 0)  
        % cuadratico
        [nC, g] = CosteL2(th, X, yTrain, r);
      else
        % Huber
        [nC, g] = CosteHuber(th, X, yTrain, fi, r); 
      end
      % Incorporar nuevo coste
      C = [C nC];    
      % Incorporar iteracion  
      I = [I i];
    until ((abs(C(i) - C(i-1))) < 0.001)
    
    fprintf('Calculo de prediccion con descenso de gradiente\n');
    
    %Calculo de las media
    medR = mean(abs(r));
    
    fprintf('Numero total de iteraciones: %d\n', i)
    fprintf('Error medio de los datos de entrenamiento %d\n\n', medR)
    
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


function [th] = descensoGradiente(XTrain, XTest, th, yTrain, ySet, t, N1, N2) 
  % falta hallar th inicial
  % Ejemplo funcion de regresión
  %Normalizar los atributos entrenamiento
  NTrain = size(XTrain,1); 
  muTrain = mean(XTrain(:,2:end)); 
  sigTrain = std(XTrain(:,2:end)); 
  XTrain(:,2:end) = (XTrain(:,2:end) - repmat(muTrain,NTrain,1))./ repmat(sigTrain,NTrain,1);
  
  % falta hallar th inicial
  % Ejemplo funcion de regresión
  %Normalizar los atributos test
  NTest = size(XTest,1); 
  muTest = mean(XTest(:,2:end)); 
  sigTest = std(XTest(:,2:end)); 
  XTest(:,2:end) = (XTest(:,2:end) - repmat(muTest,NTest,1))./ repmat(sigTest,NTest,1);
  
  % Vector de costes e iteraciones vacios
  C = [];
  I = [];
  % Decision del valor de tasa de aprendizaje alfa
  %Resolver la Regresión
  [th,i] = iterar( XTrain, th', yTrain, t, C, I , N1, N2);
  
  %Calculo del error en test
  r = XTest * th - ySet;
  medR = mean(abs(r));
  
  fprintf('Error de test con descenso de Gradiente %d\n\n', medR)
    
  %Des-Normalizarlos pesos 
  th(2:end) = th(2:end)./sigTrain'; 
  th(1)= th(1)-(muTrain*th(2:end));
  
end 


 % TRAZADO DE RESULTADOS
 
 %APARTADO 2
 
 fprintf(' ---------- Ejercicio 2 ------------------- \n\n')
 
 % regresion monovariable con datos de entrenamiento
 % ecuacion normal
 ecNormalMono(x1, x3, y1, y2, N1, N2)
 
 
 
 % APARTADO 3
 
 fprintf(' ---------- Ejercicio 3 ---------------- \n\n')
 
 % regresion multivariable con datos de entrenamiento
 % ecuacion normal
 ecNormalBiDos(x1, x2, x3, x4, y1, y2, N1, N2)
 

 % APARTADO 4
 
 fprintf(' ---------- Ejercicio 4 ---------------- \n\n')
 
 % regresion monovariable con datos de entrenamiento
 % descenso de gradiente
 MTrain = [ones(N1,1) x1];
 MTest = [ones(N2,1) x3];
 descensoGradiente(MTrain, MTest, [5000 1000], y1, y2, 0 , N1, N2)
 

 %APARTADO 5
 
 fprintf(' ---------- Ejercicio 5 ---------------- \n\n')
 
 % regresion multivariable con datos de entrenamiento
 % descenso de gradiente
 MTrain = [ones(N1,1) x1 x2];
 MTest = [ones(N2,1) x3 x4];
 descensoGradiente(MTrain, MTest, [5000 1000 50000], y1, y2, 0, N1, N2)
 

 % APARTADO 6 OPCIONAL
 
 fprintf(' ---------- Ejercicio 6 ---------------- \n\n')
 
 % regresion multivariable con datos de entrenamiento
 % coste Huber
 MTrain = [ones(N1,1) x1 x2];
 MTest = [ones(N2,1) x3 x4];
 descensoGradiente(MTrain, MTest, [5000 1000 50000], y1, y2, 1, N1 , N2)
 


 
