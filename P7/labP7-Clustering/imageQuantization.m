% Limpiar la consola
clc;

figure(1)
im = imread('smallparrot.jpg');
imshow(im)

%% datos
D = double(reshape(im,size(im,1)*size(im,2),3));

%% dimensiones
m = size(D,1);
n = size(D,2);

%% Kmeans con valor de K por defecto 16
K = 16;
ia = 0;

while (ia != K)
  %% Inicializar centroides
  semilla = randi([1 m], 1, K);
  mu0 = D(semilla, :);
  [C, ia, ic] = unique(mu0, 'rows');
end

%bucle kmeans
[mu, c, Jotas] = kmeans(D, mu0);


fprintf('Dimension final de mu\n');
size(mu)

 %% reconstruir imagen
 qIM=zeros(length(c),3);
 for h=1:K,
      ind=find(c==h);
      qIM(ind,:)=repmat(mu(h,:),length(ind),1);
 end
 qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
 
 figure(3)
 imshow(uint8(qIM));