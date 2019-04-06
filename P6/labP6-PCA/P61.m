%% Practica 6.1
% Generar datos sinteticos 2D a partir de una distribucion Gausian
% Aplicar PCA

clear all
close all

% Generar datos con myMvnRnd
% aqui se generan con uniformes
% mu = [5 3]';
% Sigma = [10 0.7; -0.7 1];
mu = [0 0]';
Sigma = [1 0.5; 0.5 1];
nsamples = 1000;
X=myMvnRnd(mu, Sigma, nsamples);
X=X';

% Centrar los datos
mu=mean(X);
cX = X-repmat(mu,size(X,1),1);

% Aplicar svd 
[U,S,v]=svd(cX);

% proyeccion
z=cX*v(:,1);

% reconstruccion
rX=z*v(:,1)' + repmat(mu,size(cX,1),1);

k=70;
rX = rX(1:k,:);

%% Dibujar el resultado
figure(100)
clf, hold on
axis equal

% mostrar los puntos 
plot(X(:,1),X(:,2),'.')

% mostrar la reconstruccion
plot(rX(:,1),rX(:,2),'k.')

% mostrar los vectores de las componentes principales en v
plot([0 v(1,1)],[0 v(1,2)],'r')
plot([0 v(2,1)],[0 v(2,2)],'g')
