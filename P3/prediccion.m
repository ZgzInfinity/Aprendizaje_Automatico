function h = prediccion(theta, X)

	h = 1./(1+exp(-(X*theta)));
    
    index_1 = find(h >= 0.5);  
	index_0 = find(h < 0.5);  
     
	h(index_1) = ones(size(index_1));  
	h(index_0) = zeros(size(index_0));
end  