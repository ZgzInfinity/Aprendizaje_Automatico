function [k, sumas, sumaK] = findValorK(vDiagMod, variabilidad)
	% Suma de los todos los K elementos de la diagonal
	kN = sum(sum(diag(vDiagMod)));
	
	% Numero de columnas de la matriz 
	N = size(vDiagMod , 2);
	
	% Control de hallazgo de mejor valor de K
	encontrado = 0;
	k = 1;

	% Vector auxiliar de las k primeras sumas
	sumas = [];

	% Bucle para recorrer matriz vDiagMod 
	while (k <= N && encontrado != 1)
	  % Valor de la suma de las k primeras componentes
	  sumaK = 0;
	  for i = 1:k
		  % Bucle de suma
		  sumaK = sumaK + vDiagMod(1 , i);
	  end
	  % Comparacion de variabilidad
	  if (sumaK / kN >= variabilidad)
		% Mejor valor de K hallado
		encontrado = 1;
	  else
		% Incrementar valor de K
		k = k + 1;
		% Incorporar la suma obtenida en iteracion actual
		sumas = [sumas ; sumaK];
	  end 
	end
end