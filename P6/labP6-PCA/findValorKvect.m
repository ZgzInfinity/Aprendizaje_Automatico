function [mejorK] = findValorKvect(vDiagMod, variabilidad)
	% Suma de los todos los K elementos de la diagonal
	kN = sum(sum(diag(vDiagMod)));
	
	% Numero de columnas de la matriz 
	N = size(vDiagMod , 2);
	
	% Control de hallazgo de mejor valor de K
	k = 1;

  ratioMax = 0;
  
	% Bucle para recorrer matriz vDiagMod 
	while (k < N)
	  % Valor de la suma de las k primeras componentes
	  sumaK = 0;
	  for i = 1:k
		  % Bucle de suma
		  sumaK = sumaK + vDiagMod(1 , i);
	  end
    
    % Calculo del ratio 
    ratio = sumaK / kN;
    
	  % Comparacion de variabilidad
	  if (ratio >= variabilidad)
      % Comprobacion de ver si el ratio es superior
      if (ratio > ratioMax)
          % Actualizacion del valor del ratio mejor
          ratioMax = ratio;
          mejorK = k;
      end
	  end 
    
    % Incrementar valor de K
    k = k + 1;
	end
end