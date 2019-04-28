function [k, sumas, sumaK] = findValorKmat(vDiagMod, variabilidad)
	% Suma de los todos los K elementos de la diagonal
	kN = sum(sum(diag(vDiagMod)));
	
	% Numero de columnas de la matriz 
	N = size(vDiagMod , 2);
	
	% Control de hallazgo de mejor valor de K
	mejorSuma = 0;
	k = 1;

	% Vector auxiliar de las k primeras sumas
	sumas = [];
  ratioMax = 0;
  
	% Bucle para recorrer matriz vDiagMod 
	while (k <= N)
	  % Valor de la suma de las k primeras componentes
	  sumaK = 0;
	  for i = 1:k
		  % Bucle de suma
		  sumaK = sumaK + vDiagMod(i , i);
	  end
	 
    % Calculo del ratio 
	  ratio = sumaK / kN;
    
		% Mejor valor de K hallado
   
    if (ratio >= variabilidad)
      % El ratio supera la variabilidad
      if (ratio >= ratioMax)
        % Se supera el maximo ratio hallado
        ratioMax = ratio;
        mejorK = k;
      end
    end
    
		% Incrementar valor de K
		k = k + 1;
		% Incorporar la suma obtenida en iteracion actual
		sumas = [sumas ; sumaK];
	end 
end