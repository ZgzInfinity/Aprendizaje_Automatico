%% P62
% Compress your own photo using SVD 
imagen=imread('foto.jpg');

ImagenGris=rgb2gray(imagen);

ImagenGris = im2double(ImagenGris);

cX = ImagenGris;

[U,S,V]=svd(cX,0);
       
error = [];

for i = 1:430
    d = U(:,1:i)*S(1:i,1:i)*V(:,1:i)';
    error = [error norm(cX-d)];
    imshow(d);
    %pause(0.1);
end

imshow(d);

pause;

plot(error);
