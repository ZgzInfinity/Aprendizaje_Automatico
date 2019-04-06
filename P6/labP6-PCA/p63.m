% load images 
% images size is 20x20. 
clear
close all

load('MNISTdata2.mat'); 

nrows=20;
ncols=20;

nimages = size(X,1);



% Show the images
% for I=1:100, 
%     imshow(reshape(X(I,:),nrows,ncols))
%     pause(0.1)
% end


%% Perform PCA over all numbers

% z should contain the projections over the first two PC
% now is just a random matrix
[U,S,V]=svd(X',0);
for i = 1:2
    d = U(:,1:i)*S(1:i,1:i)*V(:,1:i)';
end

% z=rand(size(X,1),2);

% Muestra las dos componentes principales
figure(100)
clf, hold on
plotwithcolor(d(:,1:2), y);

%% Use classifier from previous labs on the projected space







