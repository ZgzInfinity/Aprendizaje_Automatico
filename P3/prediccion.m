function h = prediccion(theta, X, umbral)

  if (nargin < 3)
    umbral = 0.5;
  end
  
	h = 1./(1+exp(-(X*theta)));
    
  h = (h >= umbral);
end  